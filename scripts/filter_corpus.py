"""Drop corpus documents whose non-Toki-Pona-letter ratio exceeds a threshold."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TP_LETTERS = set("aeijklmnopstuw")


def non_tp_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if c not in TP_LETTERS)
    return bad / len(letters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=REPO_ROOT / "data/processed/corpus.jsonl")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "data/processed/corpus.filtered.jsonl")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="drop docs whose non-TP-letter ratio exceeds this (default: 0.05)")
    args = ap.parse_args()

    kept_docs: Counter[str] = Counter()
    dropped_docs: Counter[str] = Counter()
    kept_chars: Counter[str] = Counter()
    dropped_chars: Counter[str] = Counter()

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            src = rec["source"]
            n_chars = len(rec["text"])
            if non_tp_ratio(rec["text"]) <= args.threshold:
                fout.write(line)
                kept_docs[src] += 1
                kept_chars[src] += n_chars
            else:
                dropped_docs[src] += 1
                dropped_chars[src] += n_chars

    sources = sorted(set(kept_docs) | set(dropped_docs))
    print(f"threshold: {args.threshold:.3f}")
    print(f"{'source':<10} {'kept':>8} {'dropped':>8} {'drop%':>7} "
          f"{'chars kept':>12} {'chars dropped':>14}")
    print("-" * 64)
    for src in sources:
        k, d = kept_docs[src], dropped_docs[src]
        kc, dc = kept_chars[src], dropped_chars[src]
        total = k + d
        print(f"{src:<10} {k:>8,} {d:>8,} {d/total*100:>6.1f}% "
              f"{kc:>12,} {dc:>14,}")
    print("-" * 64)
    tk, td = sum(kept_docs.values()), sum(dropped_docs.values())
    tkc, tdc = sum(kept_chars.values()), sum(dropped_chars.values())
    print(f"{'TOTAL':<10} {tk:>8,} {td:>8,} {td/(tk+td)*100:>6.1f}% "
          f"{tkc:>12,} {tdc:>14,}")
    print(f"\nwrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
