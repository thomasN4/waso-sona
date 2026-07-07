"""Throughput+quality micro-benchmark for the en→tp translate path.

Translates a fixed sample of held-out pairs under a given config and reports
wall time, sentences/s, gross tok/s, accept rate, and chrF (vs the reference
TP). Vary --concurrency / --num-predict / --no-fewshot / --url to compare
client-side trims and server parallelism (point --url at a second Ollama
instance started with OLLAMA_NUM_PARALLEL to test continuous batching).

Usage::

    python scripts/bench_translate.py --concurrency 8
    python scripts/bench_translate.py --no-fewshot --num-predict 24 --concurrency 16
    python scripts/bench_translate.py --url http://127.0.0.1:11435/api/chat --concurrency 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus as ac  # noqa: E402
from eval_translator import chrf  # noqa: E402

from sitelen import is_legal_word  # noqa: E402

DEFAULT_EVAL = REPO_ROOT / "data" / "processed" / "translation_eval.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="waso-translator")
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-predict", type=int, default=48)
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--url", default=ac.OLLAMA_CHAT_URL,
                    help="Ollama /api/chat URL (point at a 2nd instance to test parallelism)")
    args = ap.parse_args(argv)

    ac.OLLAMA_CHAT_URL = args.url  # _ollama_generate reads the module constant
    fewshot = not args.no_fewshot

    pairs = [json.loads(l) for l in
             args.eval.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]

    def work(p: dict):
        prompt = ac._translate_prompt(p["en"], fewshot=fewshot)
        try:
            text, toks = ac._ollama_generate(prompt, args.model, args.num_predict, 1.1, 64)
        except Exception:
            return None, 0, p["tp"]
        sents = ac._split_into_sentences(text)
        hyp = sents[0] if sents else ""
        return hyp, toks, p["tp"]

    t0 = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(work, pairs))
    wall = time.monotonic() - t0

    n = len(results)
    gross_tok = sum(r[1] for r in results)
    valid = sum(1 for r in results if r[0] and ac._filter_sentence(r[0])[0])
    legal = sum(1 for r in results if r[0]
                and all(is_legal_word(w) for w in ac._WORD_RE.findall(r[0])))
    chrfs = [chrf(r[0], r[2]) for r in results if r[0]]
    mean_chrf = sum(chrfs) / len(chrfs) if chrfs else 0.0

    cfg = (f"conc={args.concurrency} np={args.num_predict} "
           f"fewshot={int(fewshot)} url={args.url.split('//')[-1].split('/')[0]}")
    print(f"[{cfg}]  n={n}  wall={wall:.1f}s  "
          f"sent/s={n/wall:.1f}  tok/s={gross_tok/wall:.0f}  "
          f"valid={100*valid/n:.0f}%  legal={100*legal/n:.0f}%  "
          f"chrF={mean_chrf:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
