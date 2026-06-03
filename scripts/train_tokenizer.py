"""Train a small SentencePiece BPE tokenizer for the from-scratch TP student.

Trains on synthetic.jsonl (Stage 2 output) + a sample of the real filtered
corpus. Vocab is small (~2k) because Toki Pona's working vocabulary is tiny;
this gives ~1 token/word for known TP and byte-fallback for proper nouns.

Output: models/student_tokenizer/spm.model and spm.vocab.

Usage::

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic.jsonl"
DEFAULT_REAL = REPO_ROOT / "data" / "processed" / "corpus.filtered.jsonl"
DEFAULT_OUT = REPO_ROOT / "models" / "student_tokenizer"


def _write_training_text(synthetic: Path, real: Path, out_txt: Path,
                         real_chars_cap: int) -> tuple[int, int]:
    """Collect training text from synthetic + a chars-capped slice of the real
    corpus. The real cap keeps tokenizer training from being dominated by long
    poki documents at the expense of the conversational synthetic distribution.
    Returns (n_synthetic_lines, n_real_chars)."""
    n_syn = 0
    real_chars = 0
    with out_txt.open("w", encoding="utf-8") as out:
        if synthetic.exists():
            with synthetic.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = rec.get("text", "").strip()
                    if t:
                        out.write(t + "\n")
                        n_syn += 1
        if real.exists():
            with real.open(encoding="utf-8") as f:
                for line in f:
                    if real_chars >= real_chars_cap:
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = rec.get("text", "").strip()
                    if t:
                        out.write(t + "\n")
                        real_chars += len(t)
    return n_syn, real_chars


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    ap.add_argument("--real", type=Path, default=DEFAULT_REAL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--real-chars-cap", type=int, default=2_000_000,
                    help="cap real-corpus chars (keep balance with synthetic)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_txt = args.out_dir / "training_text.txt"
    print("Collecting training text…", flush=True)
    n_syn, real_chars = _write_training_text(
        args.synthetic, args.real, training_txt, args.real_chars_cap,
    )
    if n_syn == 0 and real_chars == 0:
        print("ERROR: no training text — check --synthetic / --real paths",
              file=sys.stderr)
        return 1
    total_chars = training_txt.stat().st_size
    print(f"  synthetic lines: {n_syn:,}", flush=True)
    print(f"  real chars:      {real_chars:,}", flush=True)
    print(f"  total chars:     {total_chars:,}", flush=True)

    print(f"Training SentencePiece BPE (vocab={args.vocab_size})…", flush=True)
    spm.SentencePieceTrainer.train(
        input=str(training_txt),
        model_prefix=str(args.out_dir / "spm"),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<bos>", eos_piece="<eos>",
        byte_fallback=True,
        normalization_rule_name="identity",  # keep TP text byte-exact
        train_extremely_large_corpus=False,
    )

    # Quick round-trip sanity check.
    sp = spm.SentencePieceProcessor(model_file=str(args.out_dir / "spm.model"))
    sample = "mi pona. tenpo suno ni la mi tawa ma kasi pi jan Ton."
    ids = sp.encode(sample, out_type=int)
    back = sp.decode(ids)
    print(f"\n  sample: {sample!r}", flush=True)
    print(f"  ids   : {ids}", flush=True)
    print(f"  decode: {back!r}", flush=True)
    print(f"  pieces: {sp.encode(sample, out_type=str)}", flush=True)
    print(f"  tokens/word ≈ {len(ids)/len(sample.split()):.2f}", flush=True)
    print(f"\n→ {args.out_dir.relative_to(REPO_ROOT)}/spm.model", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
