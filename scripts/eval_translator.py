"""Gate the en→tp translator on held-out (unseen-English) Tatoeba pairs.

Translates each held-out English sentence with the Ollama-served translator and
scores the output against the reference TP:

  * chrF   — character n-gram F2 (sacrebleu-style, self-contained), macro-mean.
  * valid  — fraction passing augment_corpus._filter_sentence (clean TP).
  * copy   — fraction that just echoed the English input (a failure mode).
  * empty  — fraction with no usable output.

The held-out English was never seen in training (split by eng_id in
build_translation_pairs.py), so this measures genuine translation, not recall.
This is the Phase 2 make-or-break gate: if chrF is near-zero / validity is poor
on unseen English, stop before any bulk run.

Usage::

    python scripts/eval_translator.py --model waso-translator
    python scripts/eval_translator.py --max-eval 500 --concurrency 4
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402

from sitelen import is_legal_word  # noqa: E402

DEFAULT_EVAL = REPO_ROOT / "data" / "processed" / "translation_eval.jsonl"


def _char_ngrams(s: str, n: int) -> collections.Counter:
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    """sacrebleu-style chrF: avg char n-gram precision/recall → F_beta, ×100."""
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for n in range(1, n_max + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p = sum(precs) / len(precs)
    r = sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def _first_sentence(text: str) -> str:
    """Take the first sentence only — the translator tends to emit a correct
    rendering and then ramble, so (like the v1 loop's [:1]/[:2]) we keep the
    leading sentence. Falls back to the first non-empty line."""
    sents = augment_corpus._split_into_sentences(text)
    if sents:
        return sents[0].strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="waso-translator")
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--max-eval", type=int, default=500,
                    help="random sample of held-out pairs to score (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=15,
                    help="how many sample translations to print")
    args = ap.parse_args(argv)

    pairs = [json.loads(ln) for ln in
             args.eval.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if args.max_eval:
        pairs = pairs[: args.max_eval]
    print(f"Scoring {len(pairs)} held-out pairs against {args.model}…", flush=True)

    def work(p: dict) -> dict:
        prompt = augment_corpus._translate_prompt(p["en"])
        try:
            text, _ = augment_corpus._ollama_generate(
                prompt, args.model, args.max_tokens, 1.1, 64)
        except Exception as exc:  # noqa: BLE001 — record, don't crash the gate
            return {"en": p["en"], "ref": p["tp"], "hyp": "", "error": str(exc)}
        hyp = _first_sentence(text)
        return {"en": p["en"], "ref": p["tp"], "hyp": hyp}

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(work, pairs):
            results.append(r)

    chrfs, valid, legal, copy, empty, errors = [], 0, 0, 0, 0, 0
    for r in results:
        hyp, ref, en = r["hyp"], r["ref"], r["en"]
        if r.get("error"):
            errors += 1
        if not hyp:
            empty += 1
            chrfs.append(0.0)
            continue
        chrfs.append(chrf(hyp, ref))
        if augment_corpus._filter_sentence(hyp)[0]:
            valid += 1
        # Word-shape signal, decomposed from `valid` (dictionary + grammar):
        # fraction whose every token is a phonotactically legal TP word-shape.
        if all(is_legal_word(w) for w in augment_corpus._WORD_RE.findall(hyp)):
            legal += 1
        if hyp.lower().strip(".!? ") == en.lower().strip(".!? "):
            copy += 1

    n = len(results)
    mean_chrf = sum(chrfs) / n if n else 0.0
    print("\n=== translator gate ===", flush=True)
    print(f"  pairs scored : {n}", flush=True)
    print(f"  chrF (macro) : {mean_chrf:.2f}", flush=True)
    print(f"  valid TP     : {valid}/{n} ({100*valid/n:.1f}%)", flush=True)
    print(f"  legal shape  : {legal}/{n} ({100*legal/n:.1f}%)", flush=True)
    print(f"  copied input : {copy}/{n} ({100*copy/n:.1f}%)", flush=True)
    print(f"  empty        : {empty}/{n} ({100*empty/n:.1f}%)", flush=True)
    if errors:
        print(f"  api errors   : {errors}", flush=True)

    print(f"\n=== {min(args.show, n)} samples (chrF | EN → HYP [REF]) ===", flush=True)
    for r in results[: args.show]:
        c = chrf(r["hyp"], r["ref"])
        print(f"  [{c:5.1f}] {r['en']}", flush=True)
        print(f"          → {r['hyp']!r}", flush=True)
        print(f"          ref: {r['ref']!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
