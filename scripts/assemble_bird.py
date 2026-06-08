"""Assemble the bird SFT training set (Phase A6).

Merges the Haiku-direct (primary, high-fidelity ambient+reactive) and translate
(supplementary topical variety) streams, injects the curated bare interjections
as ambient training examples (the gold set is held out for eval, so the
*training* set needs its own chirps), and writes `data/processed/bird.jsonl`.
Every record is re-gated through the single `bird_persona.filter_bird_line`,
deduped on `(ctx, text)`, and excluded if it collides with the held-out gold (so
train ∩ gold = ∅).

Output: data/processed/bird.jsonl — {ctx, text, mood, source}. The disjoint
held-out eval set stays in data/processed/bird_gold.jsonl.

Usage::

    python scripts/assemble_bird.py
    python scripts/assemble_bird.py --max-teacher 150 --interject-reps 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bird_persona as bp  # noqa: E402

P = REPO_ROOT / "data" / "processed"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _key(rec: dict) -> tuple[str, str]:
    return (rec.get("ctx", "").lower().strip(), rec["text"].lower().strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--haiku", type=Path, default=P / "bird_haiku.jsonl")
    ap.add_argument("--translate", type=Path, default=P / "bird_translate.jsonl")
    ap.add_argument("--gold", type=Path, default=P / "bird_gold.jsonl")
    ap.add_argument("--out", type=Path, default=P / "bird.jsonl")
    ap.add_argument("--max-haiku", type=int, default=None,
                    help="optional cap on Haiku records (default: use all)")
    ap.add_argument("--interject-reps", type=int, default=2,
                    help="inject each curated interjection N× as an ambient chirp")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    gold_keys = {_key(r) for r in _load(args.gold)}

    haiku = _load(args.haiku)
    rng.shuffle(haiku)
    if args.max_haiku:
        haiku = haiku[: args.max_haiku]
    translate = _load(args.translate)

    # Generated streams: gate + exclude gold + dedup on (ctx, text).
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dropped_gate = dropped_gold = dropped_dup = 0
    for rec in haiku + translate:
        ok, _ = bp.filter_bird_line(rec["text"])
        if rec.get("ctx"):
            ok = ok and bp.filter_bird_line(rec["ctx"])[0]
        if not ok:
            dropped_gate += 1
            continue
        k = _key(rec)
        if k in gold_keys:
            dropped_gold += 1
            continue
        if k in seen:
            dropped_dup += 1
            continue
        seen.add(k)
        kept.append({"ctx": rec.get("ctx", ""), "text": rec["text"],
                     "mood": rec.get("mood", ""), "source": rec["source"]})

    # Curated chirps are injected AFTER dedup so --interject-reps actually
    # produces copies (intentional upweighting of the bird's signature
    # behavior). They live only in training — bare interjections are a fixed
    # closed set, so holding them out for eval would be meaningless.
    for itj in sorted(bp.INTERJECTIONS):
        if ("", itj.lower().strip()) in gold_keys:
            continue
        for _ in range(args.interject_reps):
            kept.append({"ctx": "", "text": itj, "mood": "chirp",
                         "source": "interjection"})

    rng.shuffle(kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    amb = sum(1 for r in kept if not r["ctx"])
    rea = len(kept) - amb
    src = Counter(r["source"].split("/")[0] for r in kept)
    print(f"wrote {len(kept)} SFT records → {args.out}")
    print(f"  ambient {amb} ({100*amb/max(len(kept),1):.0f}%) / "
          f"reactive {rea} ({100*rea/max(len(kept),1):.0f}%)")
    print(f"  by source: {dict(src)}")
    print(f"  dropped: gate={dropped_gate} gold_overlap={dropped_gold} dup={dropped_dup}")
    print(f"  held-out gold (eval, disjoint): {len(gold_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
