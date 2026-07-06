"""Fine-tune the student into the desktop talking bird (Phase B).

Continue-trains the base deliverable (`student-8L384d-rebal-…/best.pt`) on the
bird SFT set with **reply-only loss masking** — each example is framed
`<bos> {scene}. waso li toki e ni: {reply} <eos>` (ambient: no scene) and loss is
computed only on the reply tokens (+EOS), so the model learns p(reply | scene)
and to stop after one bubble, not how to generate scenes. This is the
from-scratch analog of the teacher pipeline's `assistant_only_loss=True`.

To avoid catastrophic forgetting of general Toki Pona on ~600 persona lines (the
project's "balance is the lever" finding), each optimizer step is a coin-flip:
with prob `--bird-frac` a masked bird batch, else a raw real+translated window
(the unmasked regularizer). Two evals gate `best.pt`: the held-out **gold** masked
loss (primary, early-stop) and a **forgetting guard** — exact perplexity on the
200 held-out tatoeba docs, which must stay near the base's 2.105 nll.

Imports the model + helpers from `train_student`, the framing from
`bird_persona`, and the forgetting-guard helpers from `bench_student`.

Usage::

    python scripts/train_bird.py --dry-run     # mask unit-test, no training
    python scripts/train_bird.py --smoke        # 100-step smoke
    python scripts/train_bird.py                # full fine-tune
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_student import (  # noqa: E402
    ModelConfig, TinyTPLM, lr_at, _build_stream, sample_batch,
    _load_real_texts, _load_synthetic_texts,
)
import bird_persona as bp  # noqa: E402
from bench_student import exact_ppl, build_capped_stream, HELDOUT_N  # noqa: E402

P = REPO_ROOT / "data" / "processed"
RUNS = REPO_ROOT / "models" / "student_runs"
DEFAULT_BASE = RUNS / "student-8L384d-rebal-20260607T051752Z" / "best.pt"
DEFAULT_TOKENIZER = REPO_ROOT / "models" / "student_tokenizer" / "spm.model"
BASE_REAL_NLL = 2.105  # the base deliverable's held-out tatoeba nll (bench_student)


# ---------------------------------------------------------------------------
# Masked bird examples
# ---------------------------------------------------------------------------

def build_examples(sp, records: list[dict], bos: int, eos: int) -> list[tuple]:
    """Each record → (ids, mask_until). `ids = [bos]+encode(frame)+[eos]`; in the
    shifted targets, the first `mask_until` positions (the bos+prompt prefix) are
    masked, leaving loss on the reply tokens + EOS."""
    out = []
    for r in records:
        ctx, text = r.get("ctx", ""), r["text"]
        prompt = bp.build_prompt(ctx)
        p_ids = sp.encode(prompt, out_type=int)
        full_enc = sp.encode(bp.frame_example(ctx, text), out_type=int)
        # The SP ▁ word-boundary makes the prompt a clean token prefix; assert it,
        # else fall back to concatenating separately-encoded pieces.
        if full_enc[: len(p_ids)] != p_ids:
            full_enc = p_ids + sp.encode(text, out_type=int)
        ids = [bos] + full_enc + [eos]
        out.append((ids, len(p_ids)))
    return out


def collate(batch: list[tuple], pad_id: int, device) -> tuple:
    """(ids, mask_until) list → padded (x, y) with y=-100 on prompt + padding."""
    width = max(len(ids) for ids, _ in batch) - 1
    xs, ys = [], []
    for ids, mask_until in batch:
        x, y = ids[:-1], ids[1:]
        y = [-100] * mask_until + y[mask_until:]   # mask the prompt targets
        pad = width - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


def sample_bird_batch(exs, batch_size, pad_id, device, gen):
    idx = torch.randint(0, len(exs), (batch_size,), generator=gen).tolist()
    return collate([exs[i] for i in idx], pad_id, device)


@torch.no_grad()
def eval_bird_loss(model, exs, batch_size, pad_id, device) -> float:
    """Token-weighted masked CE over all examples (the persona objective)."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(exs), batch_size):
        x, y = collate(exs[i:i + batch_size], pad_id, device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              ignore_index=-100, reduction="sum")
        total_nll += nll.item()
        total_tok += int((y != -100).sum().item())
    model.train()
    return total_nll / max(total_tok, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--bird", type=Path, default=P / "bird.jsonl")
    ap.add_argument("--gold", type=Path, default=P / "bird_gold.jsonl")
    ap.add_argument("--real", type=Path, default=P / "corpus.filtered.jsonl")
    ap.add_argument("--translated", type=Path, default=P / "translated.jsonl")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--bird-frac", type=float, default=0.5,
                    help="prob a step draws a masked bird batch (else regularizer)")
    ap.add_argument("--reg-translated-cap", type=int, default=100_000,
                    help="cap translated sentences in the regularizer stream")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    ap.add_argument("--forget-tol", type=float, default=0.15,
                    help="warn if held-out real nll rises more than this over base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--dry-run", action="store_true",
                    help="mask unit-test only (no GPU, no training)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()

    bird_recs = [json.loads(l) for l in args.bird.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold_recs = [json.loads(l) for l in args.gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    bird_exs = build_examples(sp, bird_recs, bos, eos)
    gold_exs = build_examples(sp, gold_recs, bos, eos)

    # --- dry-run: verify the mask before touching the GPU --------------------
    if args.dry_run:
        print(f"bird={len(bird_exs)} gold={len(gold_exs)} examples")
        ok = 0
        for rec, (ids, mask_until) in list(zip(bird_recs, bird_exs))[:200]:
            y = ([-100] * mask_until + ids[1:][mask_until:])
            kept = [t for t in y if t != -100]
            decoded = sp.decode(kept).strip()
            # the kept (unmasked) targets must decode to exactly the reply
            # (EOS decodes to ''), and the prefix must be a clean token boundary.
            p_ids = sp.encode(bp.build_prompt(rec.get("ctx", "")), out_type=int)
            prefix_ok = sp.encode(bp.frame_example(rec.get("ctx", ""), rec["text"]),
                                  out_type=int)[: len(p_ids)] == p_ids
            reply_ok = decoded == rec["text"].strip()
            eos_kept = kept[-1] == eos
            if prefix_ok and reply_ok and eos_kept:
                ok += 1
            else:
                print(f"  MISMATCH prefix={prefix_ok} reply={reply_ok} eos={eos_kept}: "
                      f"{rec['text']!r} -> kept {decoded!r}")
        print(f"mask check: {ok}/{min(200,len(bird_exs))} examples clean "
              f"(reply-only, EOS unmasked, prefix token-stable)")
        print(f"  sample x→y: {bp.frame_example(bird_recs[0].get('ctx',''), bird_recs[0]['text'])!r}")
        return 0 if ok == min(200, len(bird_exs)) else 1

    if args.smoke:
        args.max_steps, args.warmup_steps, args.eval_steps = 100, 10, 50

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32

    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = RUNS / f"bird-{stamp}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))
    print(f"Run dir: {args.run_dir}", flush=True)

    # --- load base model (continue-train) ------------------------------------
    state = torch.load(args.base, map_location=device, weights_only=False)
    cfg = ModelConfig(**state["cfg"])
    model = TinyTPLM(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(state["model"])
    model.train()
    bs = cfg.block_size
    print(f"Base: {cfg.n_layer}L×{cfg.n_embd}d from {args.base.parent.name} "
          f"(base val_loss={state.get('val_loss')})", flush=True)

    # --- regularizer stream: real (held-out 200 excluded) + capped translated -
    real_all = _load_real_texts(args.real)
    real_train = real_all[HELDOUT_N:]
    trans = _load_synthetic_texts(args.translated)[: args.reg_translated_cap]
    reg_stream = _build_stream(sp, real_train + trans, bos, eos)
    print(f"Regularizer: real={len(real_train):,} + translated={len(trans):,} "
          f"→ {reg_stream.numel():,} tokens", flush=True)

    # forgetting-guard eval set: the 200 held-out tatoeba docs (exact ppl)
    real_eval_stream = build_capped_stream(sp, real_all[:HELDOUT_N], bos, eos, 1024)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=str(args.run_dir))
    bird_gen = torch.Generator().manual_seed(args.seed)
    reg_gen = torch.Generator().manual_seed(args.seed + 1)
    coin = torch.Generator().manual_seed(args.seed + 2)
    best_gold = float("inf")
    stale = 0
    t0 = time.monotonic()
    print("Fine-tuning…", flush=True)

    for step in range(args.max_steps):
        lr = lr_at(step, peak_lr=args.learning_rate, warmup_steps=args.warmup_steps,
                   max_steps=args.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        if torch.rand(1, generator=coin).item() < args.bird_frac:
            x, y = sample_bird_batch(bird_exs, args.batch_size, pad, device, bird_gen)
            kind = "bird"
        else:
            x, y = sample_batch(reg_stream, args.batch_size, bs, device, reg_gen)
            kind = "reg"
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0:
            writer.add_scalar(f"train/loss_{kind}", loss.item(), step)
            print(f"  step {step:>5}/{args.max_steps}  {kind}_loss={loss.item():.4f}  "
                  f"lr={lr:.2e}  {time.monotonic()-t0:.0f}s", flush=True)

        if step > 0 and (step % args.eval_steps == 0 or step == args.max_steps - 1):
            gold = eval_bird_loss(model, gold_exs, args.batch_size, pad, device)
            real_nll = exact_ppl(model, real_eval_stream, bs, device)["nll_per_token"]
            writer.add_scalar("eval/gold_loss", gold, step)
            writer.add_scalar("eval/real_nll", real_nll, step)
            drift = real_nll - BASE_REAL_NLL
            warn = "  ⚠FORGET" if drift > args.forget_tol else ""
            line = f"  eval @ {step}: gold={gold:.4f}  real_nll={real_nll:.3f} (Δ{drift:+.3f}){warn}"
            if gold < best_gold - args.early_stop_min_delta:
                best_gold, stale = gold, 0
                torch.save({"model": model.state_dict(), "cfg": state["cfg"],
                            "step": step, "val_loss": gold, "real_nll": real_nll},
                           args.run_dir / "best.pt")
                line += "  *saved"
            else:
                stale += 1
                line += f"  (stale {stale}/{args.early_stop_patience})"
            print(line, flush=True)
            # eyeball: one ambient + one reactive sample
            for scene in ("", "jan li sitelen e nimi mute"):
                ids = torch.tensor([[bos] + sp.encode(bp.build_prompt(scene), out_type=int)],
                                   dtype=torch.long, device=device)
                out = model.generate(ids, max_new_tokens=24, temperature=0.8, top_k=50,
                                     eos_token_id=eos)[0].tolist()
                reply = sp.decode(out[ids.shape[1]:]).strip()
                print(f"      [{scene or 'ambient'}] → {reply!r}", flush=True)
            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(f"  early stop @ {step} (best gold={best_gold:.4f})", flush=True)
                break

    torch.save({"model": model.state_dict(), "cfg": state["cfg"], "step": step,
                "val_loss": best_gold}, args.run_dir / "final.pt")
    writer.close()
    print(f"\nDone. best gold={best_gold:.4f}  → {args.run_dir}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
