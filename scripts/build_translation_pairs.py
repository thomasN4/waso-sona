"""Build English→Toki Pona parallel pairs for the Stage 2 v2 translator.

Joins Tatoeba's tok↔eng links with the tok and eng sentence exports into
aligned (English, Toki Pona) pairs, and folds in the 13 hand-aligned `lipu`
pairs (whose English side `fetch_data.iter_lipu` discards). Every TP target is
passed through `augment_corpus._filter_sentence` — the same strict validator
used at generation time — so the translator only ever learns to emit clean TP.

The held-out split is **by English sentence**: a fraction of the distinct
eng_ids is routed entirely to the eval file, so Phase 2's chrF/BLEU gate
measures translation of *unseen* English rather than memorized pairs.

The English export (`eng_sentences.tsv.bz2`) is downloaded once and read
streaming from the .bz2 (it is large and also feeds Phase 3); it is never
decompressed to disk.

Outputs (JSONL, one pair per line):
    data/processed/translation_pairs.jsonl   — train
    data/processed/translation_eval.jsonl    — held-out (by eng_id)

Usage::

    python scripts/build_translation_pairs.py
    python scripts/build_translation_pairs.py --val-frac 0.05 --seed 42
    python scripts/build_translation_pairs.py --force-download
"""
from __future__ import annotations

import argparse
import bz2
import collections
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402  (reuse the strict TP filter)

TATOEBA_DIR = REPO_ROOT / "data" / "raw" / "tatoeba"
LINKS_TSV = TATOEBA_DIR / "tok-eng_links.tsv"
TOK_TSV = TATOEBA_DIR / "tok_sentences.tsv"
ENG_BZ2 = TATOEBA_DIR / "eng_sentences.tsv.bz2"
ENG_URL = (
    "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"
)
LIPU_TSV = REPO_ROOT / "data" / "raw" / "lipu" / "translation.tsv"

DEFAULT_TRAIN_OUT = REPO_ROOT / "data" / "processed" / "translation_pairs.jsonl"
DEFAULT_EVAL_OUT = REPO_ROOT / "data" / "processed" / "translation_eval.jsonl"


def _http_download(url: str, dest: Path) -> None:
    """Mirror fetch_data._http_download — stdlib only, UA header."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "waso-sona/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _load_tok_texts() -> dict[str, str]:
    """tok_id → Toki Pona text (only rows tagged `tok`)."""
    out: dict[str, str] = {}
    with open(TOK_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "tok":
                out[parts[0]] = parts[2]
    return out


def _load_links() -> list[tuple[str, str]]:
    """List of (tok_id, eng_id) links."""
    links: list[tuple[str, str]] = []
    with open(LINKS_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                links.append((parts[0], parts[1]))
    return links


def _load_eng_texts(needed: set[str]) -> dict[str, str]:
    """Stream eng_sentences.tsv.bz2, keeping only the eng_ids we need."""
    out: dict[str, str] = {}
    with bz2.open(ENG_BZ2, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] in needed:
                out[parts[0]] = parts[2]
                if len(out) == len(needed):
                    break
    return out


def _iter_lipu_pairs() -> list[tuple[str, str, str]]:
    """The 13 hand-aligned lipu pairs as (en, tp, line_id)."""
    pairs: list[tuple[str, str, str]] = []
    if not LIPU_TSV.exists():
        return pairs
    with open(LIPU_TSV, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                tp, en = parts[0].strip(), parts[1].strip()
                pairs.append((en, tp, f"lipu:{i}"))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    ap.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of distinct eng_ids held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-download", action="store_true",
                    help="re-download eng_sentences.tsv.bz2 even if present")
    args = ap.parse_args(argv)

    for required in (LINKS_TSV, TOK_TSV):
        if not required.exists():
            print(f"missing {required} — run scripts/fetch_data.py first",
                  file=sys.stderr)
            return 1

    # 1. English export (download once, read streaming).
    if args.force_download or not ENG_BZ2.exists():
        print(f"Downloading {ENG_URL} …", flush=True)
        _http_download(ENG_URL, ENG_BZ2)
    print(f"English export: {ENG_BZ2} ({ENG_BZ2.stat().st_size/1e6:.0f} MB bz2)",
          flush=True)

    # 2. Load the join inputs.
    links = _load_links()
    tok_texts = _load_tok_texts()
    needed_eng = {eng_id for _, eng_id in links}
    print(f"links={len(links):,}  tok_sentences={len(tok_texts):,}  "
          f"distinct eng_ids needed={len(needed_eng):,}", flush=True)
    eng_texts = _load_eng_texts(needed_eng)
    print(f"english texts resolved: {len(eng_texts):,}/{len(needed_eng):,}",
          flush=True)

    # 3. Build + filter pairs. Dedup exact (en, tp). Track reject reasons.
    rejects: collections.Counter[str] = collections.Counter()
    seen_pairs: set[tuple[str, str]] = set()
    # eng_id → list of {en, tp, tok_id}
    by_eng: dict[str, list[dict]] = collections.defaultdict(list)

    for tok_id, eng_id in links:
        tp = tok_texts.get(tok_id)
        en = eng_texts.get(eng_id)
        if tp is None:
            rejects["tok_missing"] += 1
            continue
        if en is None:
            rejects["eng_missing"] += 1
            continue
        ok, reason = augment_corpus._filter_sentence(tp)
        if not ok:
            rejects[f"tp_{reason}"] += 1
            continue
        key = (en, tp)
        if key in seen_pairs:
            rejects["duplicate"] += 1
            continue
        seen_pairs.add(key)
        by_eng[eng_id].append({"en": en, "tp": tp, "tok_id": tok_id})

    tatoeba_pairs = sum(len(v) for v in by_eng.values())
    print(f"tatoeba pairs kept: {tatoeba_pairs:,} over {len(by_eng):,} eng_ids",
          flush=True)

    # 4. Held-out split by English sentence (deterministic, order-independent).
    eng_ids = sorted(by_eng)
    rng = random.Random(args.seed)
    rng.shuffle(eng_ids)
    n_val = int(len(eng_ids) * args.val_frac)
    val_ids = set(eng_ids[:n_val])

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for eng_id, recs in by_eng.items():
        bucket = eval_records if eng_id in val_ids else train_records
        for r in recs:
            bucket.append({"en": r["en"], "tp": r["tp"], "eng_id": eng_id,
                           "tok_id": r["tok_id"], "source": "tatoeba"})

    # 5. lipu pairs (no eng_id) → train only.
    lipu_kept = 0
    for en, tp, lid in _iter_lipu_pairs():
        ok, reason = augment_corpus._filter_sentence(tp)
        if not ok:
            rejects[f"tp_{reason}"] += 1
            continue
        if (en, tp) in seen_pairs:
            continue
        seen_pairs.add((en, tp))
        train_records.append({"en": en, "tp": tp, "eng_id": lid,
                              "tok_id": None, "source": "lipu"})
        lipu_kept += 1

    # 6. Write.
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 7. Report.
    print("\n=== reject reasons ===", flush=True)
    for reason, c in rejects.most_common():
        print(f"  {c:7,}  {reason}", flush=True)
    print(f"\nlipu pairs kept: {lipu_kept}", flush=True)
    print(f"train: {len(train_records):,} pairs → {args.train_out}", flush=True)
    print(f"eval : {len(eval_records):,} pairs ({n_val:,} held-out eng_ids) "
          f"→ {args.eval_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
