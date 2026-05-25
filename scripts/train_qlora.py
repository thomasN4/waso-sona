"""QLoRA fine-tune `google/gemma-4-E2B-it` on the SFT dataset.

Inputs:
    data/processed/sft_train.jsonl, sft_val.jsonl   (from build_sft_dataset.py)
Outputs:
    data/training/runs/<utc-timestamp>/
        run_config.json               # config snapshot (written before training)
        checkpoints/                  # HF Trainer-managed, save_total_limit=3
        final/                        # best PEFT adapter (≈ 100–200 MB)
        events.out.tfevents.*         # TensorBoard scalars

Notes:
- Gemma 4's stock chat template lacks the `{% generation %}` markers TRL
  needs for `assistant_only_loss=True`, so we install a minimal template
  that renders byte-identically to the stock one but exposes the markers.
  This affects only training; serving (Ollama, etc.) uses its own template.
- Mixed-precision: bf16; quantization: 4-bit nf4 (double-quant); optimizer:
  paged_adamw_8bit; gradient_checkpointing enabled with use_reentrant=False
  so PEFT input grads flow.
- Smoke mode (`--smoke`) overrides for a ~5-minute pipeline validation.

Usage::

    python scripts/train_qlora.py --check-collator   # offline mask audit
    python scripts/train_qlora.py --smoke            # 50-step end-to-end
    python scripts/train_qlora.py                    # full run
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402  (for sample-gen prompts)

DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "processed" / "sft_val.jsonl"
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "training" / "runs"
MODEL_ID = "google/gemma-4-E2B-it"

# Renders byte-identically to Gemma 4's stock chat template but exposes
# `{% generation %}` markers so TRL's assistant_only_loss masks the model
# turn correctly. Verified by comparing apply_chat_template output and
# return_assistant_tokens_mask coverage.
MINIMAL_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|turn>system\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|turn>user\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'assistant' or message['role'] == 'model' %}"
    "<|turn>model\n{% generation %}{{ message['content'] }}{% endgeneration %}<turn|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|turn>model\n"
    "{% endif %}"
)


# ---------------------------------------------------------------------------
# Tokenizer / model setup
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.chat_template = MINIMAL_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Truncate the user prompt (front) when over max_length so the assistant
    # response at the end of the sequence is preserved — otherwise loss
    # masking would silently drop the response.
    tok.truncation_side = "left"
    return tok


def load_quantized_model(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # device_map={"":0} pins everything to GPU 0; "auto" silently offloaded
    # some modules to CPU on the 8 GB card and bnb refused to dequant from
    # there. The nf4-quantized multimodal Gemma 4 e2b is ~6.7 GB resident,
    # leaving ~0.8 GB for activations + LoRA grads (fine with checkpointing).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    # Minimal kbit prep — skip peft's prepare_model_for_kbit_training because
    # it upcasts every bf16 param to fp32 for "stability", which on Gemma 4's
    # vocab=262144 means a 1.6 GB embed_tokens copy that OOMs on 8 GB VRAM.
    # bnb dequantizes nf4 → bf16 on the fly during forward; we keep
    # everything in bf16 and rely on the LoRA path for trainable params.
    for p in model.parameters():
        p.requires_grad = False
    print(f"  VRAM after load:      {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    # Move multimodal towers to CPU — we don't use them for text-only SFT
    # and they take ~260 MB of VRAM we'd rather give to logits/grads.
    if hasattr(model, "model"):
        for sub in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            mod = getattr(model.model, sub, None)
            if mod is not None:
                mod.to("cpu")
    # Disable the unused multimodal sub-projections so they don't get any
    # phantom gradient paths.
    for name, p in model.named_parameters():
        if any(k in name for k in ("vision", "audio")):
            p.requires_grad = False
    torch.cuda.empty_cache()
    print(f"  VRAM after offload:   {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model


def lora_config(r: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # The same names appear inside vision_tower and audio_tower wrapped
        # in Gemma4ClippableLinear, which PEFT does not recognize. Skip them
        # — we only fine-tune the text decoder.
        exclude_modules=r".*(vision_tower|audio_tower).*",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Collator audit — verify the chat-template override actually masks
# ---------------------------------------------------------------------------

def check_collator(tokenizer, train_path: Path, n: int = 5) -> int:
    """Audit that assistant_only_loss masking is wired up correctly.

    Returns 0 on success, 1 on failure. Run with `--check-collator` before
    training to catch a silent no-op (which would train on the few-shot
    block — exactly what we don't want)."""
    print(f"Auditing {n} examples from {train_path}…", flush=True)
    n_ok = 0
    with train_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            out = tokenizer.apply_chat_template(
                ex["messages"], tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids = out["input_ids"]
            mask = out["assistant_masks"]
            n_total = len(ids)
            n_masked = sum(mask)
            asst_content = ex["messages"][1]["content"]
            asst_only_tokens = len(tokenizer(asst_content)["input_ids"])
            # Allow ±2 token wiggle for tokenizer special-handling.
            ok = abs(n_masked - asst_only_tokens) <= 2 and 0 < n_masked < n_total
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] ex {i} ({ex['mode']:<12}): "
                  f"{n_masked}/{n_total} masked, asst-only tokens={asst_only_tokens}",
                  flush=True)
            if ok:
                n_ok += 1
    if n_ok == n:
        print(f"\n✓ all {n} examples pass — assistant_only_loss will mask correctly.",
              flush=True)
        return 0
    print(f"\n✗ {n - n_ok}/{n} examples failed the mask audit.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Sample-generation callback (off by default)
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    # Continuation prompt — verbatim from augment_corpus inference path.
    augment_corpus._continuation_prompt("mi lon tomo mi. tenpo pimeja li kama."),
    # Topic prompt — same shape as build_sft_dataset.TOPIC_PROMPTS entries.
    "Write a short Toki Pona reflection.",
    # Paraphrase prompt — verbatim from augment_corpus inference path.
    augment_corpus._paraphrase_prompt("mi pilin pona tan tenpo suno."),
]


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts: list[str], max_new_tokens: int = 128):
        self.tok = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        writer = SummaryWriter(args.logging_dir)
        was_training = model.training
        model.eval()
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                for i, prompt in enumerate(self.prompts):
                    chat = [{"role": "user", "content": prompt}]
                    input_ids = self.tok.apply_chat_template(
                        chat, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)
                    out = model.generate(
                        input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=self.tok.eos_token_id,
                    )
                    gen = out[0, input_ids.shape[1]:]
                    text = self.tok.decode(gen, skip_special_tokens=True).strip()
                    writer.add_text(f"sample/{i}", text, state.global_step)
                    writer.add_scalar(f"sample_len/{i}", len(gen), state.global_step)
        finally:
            writer.flush()
            writer.close()
            if was_training:
                model.train()
                model.enable_input_require_grads()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_run_dir(base: Path) -> Path:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    d = base / f"qlora-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="output dir; default data/training/runs/qlora-<UTC>")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="lm_head logits at vocab=262144 dominate VRAM; 256 truncates ~10%% of examples from the user side (response is preserved)")
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-epochs", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=3500)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-gen", action="store_true",
                    help="enable sample-generation callback on each eval (slower, more VRAM)")
    ap.add_argument("--smoke", action="store_true",
                    help="50-step end-to-end validation; overrides several knobs")
    ap.add_argument("--check-collator", action="store_true",
                    help="audit assistant_only_loss masking and exit")
    args = ap.parse_args(argv)

    if not args.train.exists():
        print(f"ERROR: train file not found: {args.train}", file=sys.stderr)
        return 1
    if not args.val.exists():
        print(f"ERROR: val file not found: {args.val}", file=sys.stderr)
        return 1

    # --check-collator: load tokenizer only, audit, exit.
    if args.check_collator:
        tok = load_tokenizer(args.model_id)
        return check_collator(tok, args.train)

    # Smoke-mode overrides (applied here so run_config.json records the actual values used).
    if args.smoke:
        args.max_steps = 50
        args.eval_steps = 25
        args.save_steps = 25
        args.num_epochs = 99.0
        args.sample_gen = False

    run_dir = args.run_dir or make_run_dir(DEFAULT_RUNS_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"

    # Persist config BEFORE training so it survives crashes.
    config_snapshot = {**vars(args)}
    config_snapshot = {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in config_snapshot.items()}
    (run_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2)
    )
    print(f"Run dir: {run_dir}", flush=True)

    print(f"Loading tokenizer {args.model_id}…", flush=True)
    tok = load_tokenizer(args.model_id)

    print(f"Loading model in 4-bit nf4…", flush=True)
    model = load_quantized_model(args.model_id)
    print(f"  base model loaded, dtype={model.dtype}", flush=True)

    print(f"Loading datasets…", flush=True)
    ds = load_dataset(
        "json",
        data_files={"train": str(args.train), "val": str(args.val)},
    )
    print(f"  train={len(ds['train'])}  val={len(ds['val'])}", flush=True)

    sft_config = SFTConfig(
        output_dir=str(ckpt_dir),
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        weight_decay=0.0,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=str(run_dir),
        logging_steps=10,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        packing=False,
        assistant_only_loss=True,
    )

    callbacks: list = [EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)]
    if args.sample_gen:
        callbacks.append(SampleGenerationCallback(tok, SAMPLE_PROMPTS))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,
        peft_config=lora_config(args.lora_r, args.lora_alpha),
        callbacks=callbacks,
    )

    print("Starting training…", flush=True)
    trainer.train()

    print(f"Saving final adapter to {final_dir}…", flush=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
