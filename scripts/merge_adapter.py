"""Merge a QLoRA adapter into its base model → a standalone HF checkpoint.

Prepares a fine-tuned model for GGUF conversion / Ollama serving. The adapter
targets only the attention/MLP projections (no embed/lm_head), so the merge is
shape-clean on whatever base it was trained against. The tokenizer is taken
from the adapter dir (it carries the matching vocab + the training chat
template), not the base.

Usage::

    python scripts/merge_adapter.py \
        --adapter data/training/runs/qlora-<...>/final \
        --base google/gemma-4-E2B-it \
        --out  data/training/merged/<label>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", required=True,
                    help="HF hub id or local dir of the base the adapter was trained on")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    base = args.base
    if "/" in base and not base.startswith(("google/", "meta-", "Qwen", "mistral")):
        # treat as a local path (e.g. data/training/base-pruned)
        p = Path(base)
        base = str(p if p.is_absolute() else (REPO_ROOT / p))

    print(f"loading base {base} (bf16, CPU)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    print(f"applying adapter {args.adapter}…", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("merging…", flush=True)
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"saving merged model → {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)

    # Tokenizer + chat template come from the adapter dir (correct vocab).
    tok = AutoTokenizer.from_pretrained(str(args.adapter))
    tok.save_pretrained(args.out)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
