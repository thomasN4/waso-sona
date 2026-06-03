"""Train the from-scratch tiny Toki Pona LM (Stage 3 — the deliverable).

A nanoGPT-style decoder-only transformer (~5M params default), trained on the
synthetic corpus produced by Stage 2 (data/processed/synthetic.jsonl), with an
optional held-out slice of the real corpus for a more honest val signal.

Minimal PyTorch loop — no HF Trainer / no quantization / no chat template;
from-scratch training has different needs than the QLoRA teacher pipeline.

Usage::

    python scripts/train_student.py                  # full defaults
    python scripts/train_student.py --smoke          # 100-step smoke
    python scripts/train_student.py --max-steps 50000 --n-layer 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Reduce CUDA allocator fragmentation for the small-but-frequent allocs of a
# from-scratch trainer (consistent with the other GPU scripts in this repo).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic.jsonl"
DEFAULT_REAL = REPO_ROOT / "data" / "processed" / "corpus.filtered.jsonl"
DEFAULT_TOKENIZER = REPO_ROOT / "models" / "student_tokenizer" / "spm.model"
DEFAULT_RUNS_DIR = REPO_ROOT / "models" / "student_runs"


# ---------------------------------------------------------------------------
# Model — nanoGPT-style decoder transformer, ~300 LOC if you count the loop.
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 2048
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 512
    dropout: float = 0.0
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash attention via SDPA — handles causal mask + scaling internally.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTPLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        # Init: small std for stable from-scratch training.
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                # Per nanoGPT: scale residual-path output projections.
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1) if temperature > 0 \
                else torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_token_id is not None and nxt.item() == eos_token_id:
                break
        return idx


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data — tokenize once, hold as one long stream per split, sample windows.
# ---------------------------------------------------------------------------

def _load_synthetic_texts(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("text", "").strip()
            if t:
                out.append(t)
    return out


def _load_real_texts(path: Path, max_docs: int | None = None) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("text", "").strip()
            if t:
                out.append(t)
    return out


def _build_stream(sp, texts: list[str], bos_id: int, eos_id: int) -> torch.Tensor:
    """Tokenize and concatenate texts with BOS/EOS framing per record."""
    ids: list[int] = []
    for t in texts:
        ids.append(bos_id)
        ids.extend(sp.encode(t, out_type=int))
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(stream: torch.Tensor, batch_size: int, block_size: int,
                 device: torch.device, generator: torch.Generator):
    ix = torch.randint(0, stream.size(0) - block_size - 1, (batch_size,),
                       generator=generator)
    x = torch.stack([stream[i:i + block_size] for i in ix])
    y = torch.stack([stream[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, stream, batch_size, block_size, n_batches, device, gen):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(stream, batch_size, block_size, device, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# LR schedule — warmup then cosine to 10% of peak (nanoGPT-default shape).
# ---------------------------------------------------------------------------

def lr_at(step, *, peak_lr, warmup_steps, max_steps, min_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return peak_lr * min_ratio
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_ratio + (1 - min_ratio) * cos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    ap.add_argument("--real", type=Path, default=DEFAULT_REAL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="output dir (default: models/student_runs/student-<UTC>)")
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--early-stop-patience", type=int, default=6,
                    help="stop after N consecutive evals without val improvement (0 disables)")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-3,
                    help="min syn_val decrease to count as an improvement")
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--real-eval-docs", type=int, default=200,
                    help="N docs of real corpus to use as a held-out 'honest' eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--smoke", action="store_true",
                    help="100-step smoke run with tiny config")
    args = ap.parse_args(argv)

    if args.smoke:
        args.max_steps = 100
        args.warmup_steps = 10
        args.eval_steps = 50
        args.save_steps = 100
        args.sample_steps = 100
        args.batch_size = 8
        args.n_layer = 2
        args.n_embd = 128
        args.block_size = 128

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" \
        else torch.float32

    # Run dir
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = DEFAULT_RUNS_DIR / f"student-{stamp}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.run_dir}", flush=True)
    (args.run_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}, indent=2)
    )

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    vocab_size = sp.vocab_size()
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
    print(f"Tokenizer: vocab={vocab_size} bos={bos_id} eos={eos_id} pad={pad_id}",
          flush=True)

    # Data: tokenize → stream. Split synthetic into train/val by record.
    print(f"Loading {args.synthetic}…", flush=True)
    syn_texts = _load_synthetic_texts(args.synthetic)
    print(f"  synthetic records: {len(syn_texts):,}", flush=True)
    n_val = max(1, int(len(syn_texts) * args.val_frac))
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(syn_texts), generator=rng).tolist()
    val_texts = [syn_texts[i] for i in perm[:n_val]]
    train_texts = [syn_texts[i] for i in perm[n_val:]]
    print(f"  split: train={len(train_texts):,} val={len(val_texts):,}", flush=True)

    train_stream = _build_stream(sp, train_texts, bos_id, eos_id)
    val_stream = _build_stream(sp, val_texts, bos_id, eos_id)
    print(f"  train tokens: {train_stream.numel():,}", flush=True)
    print(f"  val tokens:   {val_stream.numel():,}", flush=True)

    # Real held-out (honest eval, not from teacher's distribution).
    real_stream = None
    if args.real.exists() and args.real_eval_docs > 0:
        real_texts = _load_real_texts(args.real, args.real_eval_docs)
        real_stream = _build_stream(sp, real_texts, bos_id, eos_id)
        print(f"  real-eval tokens: {real_stream.numel():,} "
              f"({len(real_texts)} docs)", flush=True)

    # Model
    cfg = ModelConfig(
        vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, block_size=args.block_size, dropout=args.dropout,
    )
    model = TinyTPLM(cfg).to(device=device, dtype=dtype)
    nparams = n_params(model)
    print(f"Model: {nparams:,} params ({nparams/1e6:.2f}M), dtype={dtype}",
          flush=True)
    (args.run_dir / "model_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Optimizer — AdamW with weight decay only on 2-D params (nanoGPT idiom).
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad: continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    writer = SummaryWriter(log_dir=str(args.run_dir))
    sample_gen = torch.Generator().manual_seed(args.seed + 1)
    best_val = float("inf")
    stale_evals = 0
    stopped_step = None
    t0 = time.monotonic()
    print("Starting training…", flush=True)
    for step in range(args.max_steps):
        lr = lr_at(step, peak_lr=args.learning_rate,
                   warmup_steps=args.warmup_steps, max_steps=args.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = sample_batch(train_stream, args.batch_size, args.block_size,
                            device, sample_gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0:
            elapsed = time.monotonic() - t0
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            print(f"  step {step:>6}/{args.max_steps}  loss={loss.item():.4f}  "
                  f"lr={lr:.2e}  {elapsed:.0f}s", flush=True)

        if step > 0 and (step % args.eval_steps == 0 or step == args.max_steps - 1):
            val = eval_loss(model, val_stream, args.batch_size, args.block_size,
                            args.eval_batches, device, sample_gen)
            writer.add_scalar("eval/syn_loss", val, step)
            line = f"  eval @ {step}: syn_val={val:.4f}"
            if real_stream is not None and real_stream.numel() > args.block_size + 1:
                rval = eval_loss(model, real_stream, args.batch_size,
                                 args.block_size, args.eval_batches, device,
                                 sample_gen)
                writer.add_scalar("eval/real_loss", rval, step)
                line += f"  real_val={rval:.4f}"
            if val < best_val - args.early_stop_min_delta:
                best_val = val
                stale_evals = 0
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                            "step": step, "val_loss": val},
                           args.run_dir / "best.pt")
            else:
                stale_evals += 1
                line += f"  (stale {stale_evals}/{args.early_stop_patience})"
            print(line, flush=True)
            if args.early_stop_patience > 0 and stale_evals >= args.early_stop_patience:
                stopped_step = step
                print(f"  early stop @ {step}: no syn_val improvement "
                      f"for {stale_evals} evals (best={best_val:.4f})", flush=True)
                break

        if step > 0 and step % args.sample_steps == 0:
            ctx = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            out = model.generate(ctx, max_new_tokens=60, temperature=0.8,
                                 top_k=50, eos_token_id=eos_id)
            text = sp.decode(out[0].tolist())
            writer.add_text("sample", text, step)
            print(f"    sample: {text!r}", flush=True)

        if step > 0 and step % args.save_steps == 0:
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "step": step},
                       args.run_dir / "last.pt")

    # Final save
    final_step = stopped_step if stopped_step is not None else args.max_steps
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                "step": final_step, "val_loss": best_val},
               args.run_dir / "final.pt")
    writer.close()
    reason = f"early stop @ {stopped_step}" if stopped_step is not None \
        else f"reached max_steps ({args.max_steps})"
    print(f"\nDone ({reason}). best syn_val={best_val:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
