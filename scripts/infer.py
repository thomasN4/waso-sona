"""Generate from the QLoRA fine-tuned Gemma 4 adapter.

By default loads the most recent run's `final/` adapter from
`models/runs/qlora-*`, wraps a seed sentence in the same
`_continuation_prompt` shape used during SFT, and prints the generated
text. With `--compare-base`, generates with the adapter disabled too so
you can see what the fine-tune actually changed.

Requires KDE/Wayland killed so the base model + adapter fit in VRAM —
see PIPELINE.md.

Usage::

    # Use the latest run's adapter, default seed + continuation mode
    python scripts/infer.py

    # Custom seed
    python scripts/infer.py --prompt "tenpo suno ni la mi tawa ma kasi."

    # Side-by-side base vs adapter on the same prompt (same seed)
    python scripts/infer.py --compare-base

    # Pin a specific adapter
    python scripts/infer.py --adapter models/runs/qlora-20260526T203942Z/final
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402  (for _continuation_prompt + _paraphrase_prompt)

DEFAULT_RUNS_DIR = REPO_ROOT / "models" / "runs"
DEFAULT_BASE_MODEL = "google/gemma-4-E2B-it"
DEFAULT_SEED_SENTENCE = "mi lon tomo mi. tenpo pimeja li kama."


def latest_adapter_dir() -> Path:
    """Most recent run dir whose final/adapter_model.safetensors exists."""
    candidates = sorted(DEFAULT_RUNS_DIR.glob("qlora-*"), reverse=True)
    for run in candidates:
        if (run / "final" / "adapter_model.safetensors").exists():
            return run / "final"
    raise FileNotFoundError(
        f"no final adapter found under {DEFAULT_RUNS_DIR}; "
        "run scripts/train_qlora.py first"
    )


def build_user_message(mode: str, seed: str) -> str:
    if mode == "continuation":
        return augment_corpus._continuation_prompt(seed)
    if mode == "paraphrase":
        return augment_corpus._paraphrase_prompt(seed)
    return seed  # raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", type=Path, default=None,
                    help="path to a PEFT adapter dir (default: most recent run's final/)")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_SEED_SENTENCE,
                    help="seed sentence (used as-is in --mode raw, wrapped otherwise)")
    ap.add_argument("--mode", choices=["continuation", "paraphrase", "raw"],
                    default="continuation",
                    help="how to wrap the seed before sending it to the model")
    ap.add_argument("--compare-base", action="store_true",
                    help="also generate with the adapter disabled, same seed, side-by-side")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42,
                    help="generation seed; reused for base+adapter so they're comparable")
    args = ap.parse_args(argv)

    adapter_dir = args.adapter or latest_adapter_dir()
    print(f"adapter: {adapter_dir.relative_to(REPO_ROOT) if adapter_dir.is_relative_to(REPO_ROOT) else adapter_dir}", flush=True)
    print(f"base:    {args.base_model}", flush=True)

    # Tokenizer comes from the adapter dir — picks up our minimal chat
    # template with the {% generation %} markers, matching training.
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("loading base model in 4-bit nf4…", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    # Move multimodal towers to CPU — same as training (we don't use them).
    if hasattr(model, "model"):
        for sub in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            mod = getattr(model.model, sub, None)
            if mod is not None:
                mod.to("cpu")
    torch.cuda.empty_cache()

    print(f"applying LoRA adapter from {adapter_dir.name}…", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    user_msg = build_user_message(args.mode, args.prompt)
    chat = [{"role": "user", "content": user_msg}]
    encoded = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    # After moving vision/audio towers to CPU, model.device can point to CPU
    # (it picks up the first param). Pin to the text embedding's actual device.
    gen_device = model.get_input_embeddings().weight.device
    # Newer transformers may return a BatchEncoding instead of a bare tensor.
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(gen_device)
    else:
        input_ids = encoded.to(gen_device)

    def generate() -> str:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    print()
    print("=" * 70)
    print(f"seed:  {args.prompt}")
    print(f"mode:  {args.mode}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    adapter_out = generate()
    print("\n--- ADAPTER ---")
    print(adapter_out)

    if args.compare_base:
        torch.manual_seed(args.seed)
        with model.disable_adapter():
            base_out = generate()
        print("\n--- BASE (no adapter) ---")
        print(base_out)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
