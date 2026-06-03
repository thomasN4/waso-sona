"""Inference for the from-scratch Toki Pona student LM.

Loads a checkpoint + tokenizer, generates from an optional Latin-TP prompt
(or just BOS), and optionally renders the Latin output as sitelen pona UCSUR
via the application-layer translator.

Usage::

    python scripts/talk_to_student.py
    python scripts/talk_to_student.py --prompt "tenpo suno ni la" --ucsur
    python scripts/talk_to_student.py --checkpoint models/student_runs/student-<UTC>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_student import ModelConfig, TinyTPLM  # noqa: E402

DEFAULT_TOKENIZER = REPO_ROOT / "models" / "student_tokenizer" / "spm.model"
DEFAULT_RUNS = REPO_ROOT / "models" / "student_runs"


def latest_checkpoint() -> Path:
    runs = sorted(DEFAULT_RUNS.glob("student-*"), reverse=True)
    for r in runs:
        for name in ("best.pt", "final.pt", "last.pt"):
            if (r / name).exists():
                return r / name
    raise FileNotFoundError(
        f"no student checkpoint under {DEFAULT_RUNS}; "
        "run scripts/train_student.py first"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to .pt (default: best.pt from newest student run)")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompt", default="",
                    help="Latin-TP prompt; empty = generate from BOS only")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1, help="how many samples")
    ap.add_argument("--ucsur", action="store_true",
                    help="also print the sitelen pona UCSUR rendering")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt_path = args.checkpoint or latest_checkpoint()
    print(f"checkpoint: {ckpt_path.relative_to(REPO_ROOT) if ckpt_path.is_relative_to(REPO_ROOT) else ckpt_path}",
          flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    bos_id, eos_id = sp.bos_id(), sp.eos_id()

    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**state["cfg"])
    model = TinyTPLM(cfg).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"model: {cfg.n_layer}L × {cfg.n_embd}d, vocab={cfg.vocab_size}, "
          f"ctx={cfg.block_size}", flush=True)
    if "val_loss" in state:
        print(f"        val_loss={state['val_loss']:.4f} @ step {state.get('step')}",
              flush=True)

    # Build the prompt: always start with BOS; append encoded prompt tokens.
    prompt_ids = [bos_id]
    if args.prompt:
        prompt_ids.extend(sp.encode(args.prompt, out_type=int))

    sitelen_translate = None
    if args.ucsur:
        from sitelen.translate import latin_to_ucsur  # noqa: E402
        sitelen_translate = latin_to_ucsur

    for i in range(args.n):
        torch.manual_seed(args.seed + i)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             eos_token_id=eos_id)
        # Drop the BOS in the printed output.
        gen_ids = out[0].tolist()
        if gen_ids and gen_ids[0] == bos_id:
            gen_ids = gen_ids[1:]
        text = sp.decode(gen_ids).strip()
        print(f"\n--- sample {i+1} ---")
        print(text)
        if sitelen_translate is not None:
            try:
                print(f"\nUCSUR:\n{sitelen_translate(text)}")
            except Exception as e:
                print(f"(sitelen render failed: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
