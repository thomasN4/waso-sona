"""Build Toki Pona paraphrase pairs from Tatoeba translation links.

Two TP sentences that share the same English translation are mutual
paraphrases. This downloads the `tok-eng` link export, groups TP sentences by
their shared English hub, and writes the ordered (bidirectional) pairs for
use as a `paraphrase` SFT mode in build_sft_dataset.py.

Pure stdlib. Usage::

    python scripts/build_paraphrase_pairs.py
    python scripts/build_paraphrase_pairs.py --min-cluster 2
"""
from __future__ import annotations

import argparse
import bz2
import collections
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOK_SENTENCES = REPO_ROOT / "data" / "raw" / "tatoeba" / "tok_sentences.tsv"
LINKS_URL = (
    "https://downloads.tatoeba.org/exports/per_language/tok/tok-eng_links.tsv.bz2"
)
LINKS_BZ2 = REPO_ROOT / "data" / "raw" / "tatoeba" / "tok-eng_links.tsv.bz2"
LINKS_TSV = REPO_ROOT / "data" / "raw" / "tatoeba" / "tok-eng_links.tsv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "paraphrase_pairs.jsonl"


def _download_links(force: bool) -> None:
    if LINKS_TSV.exists() and not force:
        return
    LINKS_BZ2.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {LINKS_URL}…", flush=True)
    req = urllib.request.Request(LINKS_URL, headers={"User-Agent": "waso-sona/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(LINKS_BZ2, "wb") as f:
        shutil.copyfileobj(resp, f)
    with bz2.open(LINKS_BZ2, "rb") as src, open(LINKS_TSV, "wb") as dst:
        shutil.copyfileobj(src, dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-cluster", type=int, default=2,
                    help="min distinct TP sentences per English hub (default 2)")
    ap.add_argument("--force", action="store_true", help="re-download the links")
    args = ap.parse_args(argv)

    if not TOK_SENTENCES.exists():
        print(f"ERROR: {TOK_SENTENCES} missing — run fetch_data.py first",
              file=sys.stderr)
        return 1

    _download_links(args.force)

    tok_text: dict[str, str] = {}
    with TOK_SENTENCES.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                tok_text[p[0]] = p[2]

    eng_to_tok: dict[str, set[str]] = collections.defaultdict(set)
    with LINKS_TSV.open(encoding="utf-8") as f:
        for line in f:
            t, e = line.rstrip("\n").split("\t")
            if t in tok_text:
                eng_to_tok[e].add(t)

    n_clusters = 0
    n_pairs = 0
    size_hist: collections.Counter = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for eng_id, toks in eng_to_tok.items():
            texts = sorted({tok_text[t].strip() for t in toks})
            if len(texts) < args.min_cluster:
                continue
            n_clusters += 1
            size_hist[len(texts)] += 1
            # All ordered pairs (bidirectional): teach "given any cluster
            # member, produce another".
            for a in texts:
                for b in texts:
                    if a == b:
                        continue
                    out.write(json.dumps({"a": a, "b": b, "eng_id": eng_id},
                                         ensure_ascii=False) + "\n")
                    n_pairs += 1

    print(f"TP sentences:        {len(tok_text):,}")
    print(f"Paraphrase clusters: {n_clusters:,}")
    print(f"  size histogram:    {dict(sorted(size_hist.items()))}")
    print(f"Ordered pairs:       {n_pairs:,}")
    print(f"→ {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
