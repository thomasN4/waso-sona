"""Drop corpus documents whose non-Toki-Pona-letter ratio exceeds a threshold.

Optionally also run a relaxed sentence-quality check on each doc and drop docs
whose sentence-level pass rate is below --min-sentence-pass-rate. The check is
a targeted subset of augment_corpus._filter_sentence — only the structural-
error rules (unknown lowercase word, double `li`, `li e`, leading `li`,
high repetition). Length and missing-predicate rules are intentionally
omitted: they were calibrated for generator output (3-40 words) and would
falsely reject valid short corpus sentences like "mi suli.".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_corpus import TP_VOCAB, _WORD_RE  # noqa: E402

TP_LETTERS = set("aeijklmnopstuw")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Older / dialect Toki Pona words not present in the modern UCSUR vocab but
# attested across the real corpora. Treating these as known prevents valid
# historical content from being rejected as "unknown_word". Counts in the
# raw corpus (as of 2026-05-27): ali=4818, majuna=406, powe=370, po=268.
PERMISSIVE_EXTRA_VOCAB = frozenset({"ali", "powe", "majuna", "po"})
ALLOWED_VOCAB = TP_VOCAB | PERMISSIVE_EXTRA_VOCAB


def non_tp_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if c not in TP_LETTERS)
    return bad / len(letters)


def quality_check(sentence: str) -> tuple[bool, str]:
    """Targeted subset of augment_corpus._filter_sentence for real corpus text.

    Returns (ok, reason). Sentences with no word tokens are skipped (ok=True,
    reason="skip") so callers can exclude them from the pass-rate denominator.
    Skips length and missing-predicate checks; keeps unknown-word, double/leading
    particle errors, and repetition caps.
    """
    words = _WORD_RE.findall(sentence)
    if not words:
        return True, "skip"
    lower_words = [w.lower() for w in words]

    # Unknown lowercase word (uppercase passes as proper noun).
    for orig, lower in zip(words, lower_words):
        if orig[0].isupper():
            continue
        if lower not in ALLOWED_VOCAB:
            return False, "unknown_word"

    text_lc = " " + sentence.lower() + " "
    if re.search(r"\bli\s+li\b", text_lc):
        return False, "double_li"
    if re.search(r"\bli\s+e\b", text_lc):
        return False, "li_e_error"
    if re.match(r"\s*li\b", sentence.lower()):
        return False, "leading_li"

    n = len(lower_words)
    counts: Counter[str] = Counter(lower_words)
    if n >= 5 and max(counts.values()) / n > 0.4:
        return False, "unigram_repetition"
    if n >= 4:
        bigrams = list(zip(lower_words, lower_words[1:]))
        bg_counts = Counter(bigrams)
        if bg_counts and max(bg_counts.values()) > 2:
            return False, "bigram_repetition"

    return True, ""


def sentence_pass_rate(text: str) -> tuple[float, Counter[str]]:
    """Fraction of *word-bearing* sentences in `text` that pass quality_check,
    along with a counter of rejection reasons (for diagnostics)."""
    sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    reasons: Counter[str] = Counter()
    passed = considered = 0
    for s in sents:
        ok, reason = quality_check(s)
        if reason == "skip":
            continue
        considered += 1
        if ok:
            passed += 1
        else:
            reasons[reason] += 1
    if considered == 0:
        return 1.0, reasons  # nothing to judge → don't penalize
    return passed / considered, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path,
                    default=REPO_ROOT / "data/processed/corpus.jsonl")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "data/processed/corpus.filtered.jsonl")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="drop docs whose non-TP-letter ratio exceeds this (default: 0.05)")
    ap.add_argument("--min-sentence-pass-rate", type=float, default=None,
                    help="if set, additionally drop docs whose fraction of sentences "
                         "passing augment_corpus._filter_sentence is below this "
                         "(e.g. 0.5). Off by default — letter-ratio only.")
    args = ap.parse_args()

    kept_docs: Counter[str] = Counter()
    dropped_docs: Counter[str] = Counter()
    kept_chars: Counter[str] = Counter()
    dropped_chars: Counter[str] = Counter()
    dropped_by_letter: Counter[str] = Counter()
    dropped_by_sentence: Counter[str] = Counter()
    sentence_reject_reasons: Counter[str] = Counter()

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            src = rec["source"]
            n_chars = len(rec["text"])
            keep = non_tp_ratio(rec["text"]) <= args.threshold
            if not keep:
                dropped_by_letter[src] += 1
            elif args.min_sentence_pass_rate is not None:
                rate, reasons = sentence_pass_rate(rec["text"])
                if rate < args.min_sentence_pass_rate:
                    keep = False
                    dropped_by_sentence[src] += 1
                    sentence_reject_reasons.update(reasons)
            if keep:
                fout.write(line)
                kept_docs[src] += 1
                kept_chars[src] += n_chars
            else:
                dropped_docs[src] += 1
                dropped_chars[src] += n_chars

    sources = sorted(set(kept_docs) | set(dropped_docs))
    print(f"letter-ratio threshold: {args.threshold:.3f}")
    if args.min_sentence_pass_rate is not None:
        print(f"min sentence pass rate: {args.min_sentence_pass_rate:.3f}")
    print(f"{'source':<10} {'kept':>8} {'drop(L)':>8} {'drop(S)':>8} "
          f"{'drop%':>7} {'chars kept':>12} {'chars dropped':>14}")
    print("-" * 78)
    for src in sources:
        k, d = kept_docs[src], dropped_docs[src]
        kc, dc = kept_chars[src], dropped_chars[src]
        total = k + d
        print(f"{src:<10} {k:>8,} {dropped_by_letter[src]:>8,} "
              f"{dropped_by_sentence[src]:>8,} {d/total*100:>6.1f}% "
              f"{kc:>12,} {dc:>14,}")
    print("-" * 78)
    tk, td = sum(kept_docs.values()), sum(dropped_docs.values())
    tkc, tdc = sum(kept_chars.values()), sum(dropped_chars.values())
    tdl, tds = sum(dropped_by_letter.values()), sum(dropped_by_sentence.values())
    print(f"{'TOTAL':<10} {tk:>8,} {tdl:>8,} {tds:>8,} {td/(tk+td)*100:>6.1f}% "
          f"{tkc:>12,} {tdc:>14,}")
    if sentence_reject_reasons:
        print("\nSentence-level rejection reasons (across dropped docs):")
        for reason, count in sentence_reject_reasons.most_common():
            print(f"  {reason:<22} {count:>8,}")
    try:
        out_label = args.output.relative_to(REPO_ROOT)
    except ValueError:
        out_label = args.output
    print(f"\nwrote {out_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
