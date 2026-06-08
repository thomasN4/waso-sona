"""Inference for the desktop talking bird (Phase B) — and the app seam.

Loads a fine-tuned bird checkpoint, feeds the cue (`waso li toki e ni:`,
optionally after a TP scene), samples one short reply, strips the cue, and prints
the Latin TP — or, with `--ucsur`, the sitelen pona rendering. `--loop` emits one
bubble per `--interval` seconds in the `text⇥seconds` format the desktop-bird
reads on stdin, so the deliverable hookup is::

    python scripts/talk_to_bird.py --loop --ucsur | cargo run --release   # in desktop-bird/

`--scene "<tp>"` drives a reactive comment; otherwise it free-chirps (ambient).
In `--loop` with no fixed `--scene`, it alternates ambient chirps with random
`bird_persona.SCENES` reactions to show the conditioned behavior.

Usage::

    python scripts/talk_to_bird.py --n 8                       # ambient samples
    python scripts/talk_to_bird.py --scene "jan li sitelen e nimi mute" --ucsur
    python scripts/talk_to_bird.py --loop --interval 5 --ucsur | cargo run
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_student import ModelConfig, TinyTPLM  # noqa: E402
import bird_persona as bp  # noqa: E402

RUNS = REPO_ROOT / "models" / "student_runs"
DEFAULT_TOKENIZER = REPO_ROOT / "models" / "student_tokenizer" / "spm.model"


def latest_bird_checkpoint() -> Path:
    runs = sorted(RUNS.glob("bird-*"), reverse=True)
    for r in runs:
        for name in ("best.pt", "final.pt"):
            if (r / name).exists():
                return r / name
    raise FileNotFoundError(f"no bird checkpoint under {RUNS}; run train_bird.py first")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--scene", default=None, help="TP scene for a reactive reply")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--n", type=int, default=6, help="samples (non-loop mode)")
    ap.add_argument("--ucsur", action="store_true", help="render sitelen pona UCSUR")
    ap.add_argument("--loop", action="store_true", help="emit a bubble every --interval s")
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or latest_bird_checkpoint()
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    bos, eos = sp.bos_id(), sp.eos_id()
    state = torch.load(ckpt, map_location=args.device, weights_only=False)
    model = TinyTPLM(ModelConfig(**state["cfg"])).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    latin_to_ucsur = None
    if args.ucsur:
        from sitelen.translate import latin_to_ucsur as _l2u
        latin_to_ucsur = _l2u

    scene_pool = [s for lst in bp.SCENES.values() for s in lst]
    rng = random.Random(args.seed)

    def chirp(scene: str, gen_seed: int) -> str:
        torch.manual_seed(gen_seed)
        prompt_ids = sp.encode(bp.build_prompt(scene), out_type=int)
        idx = torch.tensor([[bos] + prompt_ids], dtype=torch.long, device=args.device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             eos_token_id=eos)[0].tolist()
        return sp.decode(out[idx.shape[1]:]).strip()

    def emit(text: str) -> str | None:
        if not text:
            return None
        if latin_to_ucsur is None:
            return text
        try:
            return latin_to_ucsur(text)
        except Exception as e:  # noqa: BLE001
            print(f"(sitelen render failed: {e})", file=sys.stderr)
            return None

    if args.loop:
        # The app seam: one `text⇥seconds` line per interval on stdout.
        i = 0
        while True:
            scene = args.scene if args.scene is not None else (
                "" if i % 2 == 0 else rng.choice(scene_pool))
            text = chirp(scene, args.seed + i)
            rendered = emit(text)
            if rendered:
                sys.stdout.write(f"{rendered}\t{args.interval:g}\n")
                sys.stdout.flush()
            i += 1
            time.sleep(args.interval)
        return 0

    print(f"checkpoint: {ckpt.relative_to(REPO_ROOT)}  "
          f"(gold val_loss={state.get('val_loss'):.4f})", file=sys.stderr)
    for i in range(args.n):
        scene = args.scene if args.scene is not None else ""
        text = chirp(scene, args.seed + i)
        tag = f"[{scene}] " if scene else ""
        print(f"{tag}{text}")
        if latin_to_ucsur is not None:
            print(f"  UCSUR: {emit(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
