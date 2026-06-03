"""Build the English source pool to translate into Toki Pona (Stage 2 v2, P3).

Toki Pona can only express simple, concrete ideas, so this collects *simple*
English sentences from two sources and pre-filters them — abstract / technical
English would just translate into unknown words and get dropped by the
translator's filter, wasting teacher time.

Sources:
  A. **Tatoeba English without a TP link** — the English sentences NOT already
     paired with a TP sentence (those are the parallel data; translating them
     adds nothing new). Same register the translator trained on, huge volume.
  B. **Simple-English Wikipedia** — concrete encyclopedic prose, for topical
     range. Fetched as a dump and wikitext-stripped (reusing fetch_data).

Simple/concrete filter (precision over recall — millions available):
  * 3..--max-words words, ASCII only, ends with . ! ?
  * no very long tokens (jargon), ≤2 mid-sentence capitalized words (proper
    nouns → transliteration soup), few digits
  * ≥ --min-common-ratio of words in the COMMON set (top --common-vocab-size
    words by frequency over all Tatoeba English) — keeps plain, common English.

Output: data/processed/english_source.txt (one sentence per line, shuffled).

Usage::

    python scripts/build_english_source.py                      # both sources
    python scripts/build_english_source.py --skip-simplewiki    # Tatoeba only
    python scripts/build_english_source.py --max-sentences 300000
"""
from __future__ import annotations

import argparse
import bz2
import collections
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_data import _http_download, _strip_wikitext  # noqa: E402

TATOEBA_DIR = REPO_ROOT / "data" / "raw" / "tatoeba"
LINKS_TSV = TATOEBA_DIR / "tok-eng_links.tsv"
ENG_BZ2 = TATOEBA_DIR / "eng_sentences.tsv.bz2"

SIMPLEWIKI_DIR = REPO_ROOT / "data" / "raw" / "simplewiki"
SIMPLEWIKI_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)
SIMPLEWIKI_BZ2 = SIMPLEWIKI_DIR / "simplewiki-latest-pages-articles.xml.bz2"

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "english_source.txt"

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _linked_eng_ids() -> set[str]:
    ids: set[str] = set()
    with open(LINKS_TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def _stream_eng(only_unlinked: set[str] | None):
    """Yield (id, text) from eng_sentences.tsv.bz2. If only_unlinked is given,
    yield only rows whose id is NOT in it."""
    with bz2.open(ENG_BZ2, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if only_unlinked is not None and parts[0] in only_unlinked:
                continue
            yield parts[0], parts[2]


def _iter_simplewiki_sentences():
    """Sentence-split bodies of main-namespace Simple-English Wikipedia pages."""
    if not SIMPLEWIKI_BZ2.exists():
        SIMPLEWIKI_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SIMPLEWIKI_URL} …", flush=True)
        _http_download(SIMPLEWIKI_URL, SIMPLEWIKI_BZ2)
    ns_re = re.compile(r"^\{[^}]+\}")
    title = ns = None
    with bz2.open(SIMPLEWIKI_BZ2, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            tag = ns_re.sub("", elem.tag)
            if tag == "title":
                title = (elem.text or "").strip()
            elif tag == "ns":
                ns = (elem.text or "").strip()
            elif tag == "text":
                if ns == "0" and title and elem.text \
                        and not elem.text.lstrip().upper().startswith("#REDIRECT"):
                    body = _strip_wikitext(elem.text)
                    for sent in _SENT_SPLIT_RE.split(body):
                        sent = sent.strip()
                        if sent:
                            yield sent
            elif tag == "page":
                title = ns = None
                elem.clear()


def is_simple(sentence: str, common: frozenset[str],
              max_words: int, min_common_ratio: float) -> bool:
    s = sentence.strip()
    if not s or s[-1] not in ".!?" or not s.isascii():
        return False
    # Wikitext-strip residue: removed links/templates leave double spaces,
    # dangling " ." / " ," and stray single letters ("were d from their men").
    if "  " in s or " ." in s or " ," in s:
        return False
    toks = _WORD_RE.findall(s)
    n = len(toks)
    if n < 3 or n > max_words:
        return False
    if any(len(t) > 13 for t in toks):
        return False
    if any(len(t) == 1 and t.lower() not in ("a", "i") for t in toks):
        return False
    if sum(1 for t in toks[1:] if t[:1].isupper()) > 2:
        return False
    if sum(c.isdigit() for c in s) > 4:
        return False
    lower = [t.lower() for t in toks]
    if sum(1 for t in lower if t in common) / n < min_common_ratio:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-sentences", type=int, default=300000,
                    help="cap on total kept sentences (0 = no cap)")
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--min-common-ratio", type=float, default=0.75)
    ap.add_argument("--common-vocab-size", type=int, default=4000)
    ap.add_argument("--skip-tatoeba", action="store_true")
    ap.add_argument("--skip-simplewiki", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not ENG_BZ2.exists():
        print(f"missing {ENG_BZ2} — run build_translation_pairs.py first",
              file=sys.stderr)
        return 1

    # 1. COMMON vocab — word frequency over ALL Tatoeba English (one stream).
    print("Building common-word vocab from Tatoeba English…", flush=True)
    freq: collections.Counter[str] = collections.Counter()
    for _id, text in _stream_eng(None):
        freq.update(t.lower() for t in _WORD_RE.findall(text))
    common = frozenset(w for w, _ in freq.most_common(args.common_vocab_size))
    print(f"  vocab={len(freq):,} word-forms → COMMON top {len(common):,}",
          flush=True)

    kept: list[str] = []
    per_source: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()

    def consider(sent: str, source: str) -> None:
        if not is_simple(sent, common, args.max_words, args.min_common_ratio):
            return
        key = sent.lower()
        if key in seen:
            return
        seen.add(key)
        kept.append(sent)
        per_source[source] += 1

    # 2. Pass A — Tatoeba English without a TP link.
    if not args.skip_tatoeba:
        linked = _linked_eng_ids()
        print(f"Pass A: Tatoeba English (excluding {len(linked):,} linked)…",
              flush=True)
        seen_a = scanned_a = 0
        for _id, text in _stream_eng(linked):
            scanned_a += 1
            consider(text, "tatoeba")
        print(f"  scanned {scanned_a:,} → kept {per_source['tatoeba']:,}",
              flush=True)

    # 3. Pass B — Simple-English Wikipedia.
    if not args.skip_simplewiki:
        print("Pass B: Simple-English Wikipedia…", flush=True)
        scanned_b = 0
        for sent in _iter_simplewiki_sentences():
            scanned_b += 1
            consider(sent, "simplewiki")
        print(f"  scanned {scanned_b:,} sentences → kept "
              f"{per_source['simplewiki']:,}", flush=True)

    # 4. Shuffle + cap + write.
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.max_sentences and len(kept) > args.max_sentences:
        print(f"capping {len(kept):,} → {args.max_sentences:,}", flush=True)
        kept = kept[: args.max_sentences]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in kept:
            f.write(s + "\n")

    print(f"\nkept by source: {dict(per_source)}", flush=True)
    print(f"total written : {len(kept):,} → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
