"""Translate English source sentences into Toki Pona (Stage 2 v2, Phase 4).

The content lever: each line of data/processed/english_source.txt
(build_english_source.py) is translated by the Ollama-served `waso-translator`
into one TP sentence. Unlike paraphrase/continuation (which only recombine
existing TP), the novelty here enters from the *English source*, so the corpus
gains genuine topical range.

Reuses augment_corpus wholesale: `_ollama_generate` (chat endpoint, so the
translator's chat template wraps the prompt), `_translate_prompt`,
`_split_into_sentences`, `_filter_sentence`, and the resumable `.done` sidecar +
global cross-run dedup pattern. The translator translates correctly then tends
to ramble, so we keep only the FIRST sentence (like the v1 [:1]/[:2] trims).

Translation-specific rejects: `empty` (no output) and `copied_source` (echoed
the English). Leftover English shows up as `_filter_sentence`'s `unknown_word`.

Output: data/processed/translated.jsonl (appended) — {source, eng, text}.

Defaults are tuned from bench_translate.py: no few-shot + num_predict 24 +
concurrency 8. The big throughput lever is the *server* — set
`OLLAMA_NUM_PARALLEL` (see scripts/enable_ollama_parallel.sh) so Ollama
continuously batches; otherwise it serializes and concurrency does nothing.

Usage::

    python scripts/translate_corpus.py --max-sentences 2000   # pilot
    python scripts/translate_corpus.py --concurrency 16        # bulk (batched server)
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
import time
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus as ac  # noqa: E402  (reuse Ollama + filter + helpers)

DEFAULT_MODEL = "waso-translator"
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "english_source.txt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "translated.jsonl"


def _translate_one(
    eng: str, model: str, max_tokens: int, repeat_penalty: float,
    repeat_last_n: int, fewshot: bool,
) -> tuple[dict | None, str, int]:
    """Translate one English sentence. Returns (record_or_None, reason, tokens)."""
    text, toks = ac._ollama_generate(
        ac._translate_prompt(eng, fewshot=fewshot), model, max_tokens,
        repeat_penalty, repeat_last_n)
    sents = ac._split_into_sentences(text)
    if not sents:
        return None, "empty", toks
    hyp = sents[0]
    if hyp.lower().strip(".!? ") == eng.lower().strip(".!? "):
        return None, "copied_source", toks
    ok, reason = ac._filter_sentence(hyp)
    if not ok:
        return None, reason, toks
    return {"source": f"{model}/translate", "eng": eng, "text": hyp}, "", toks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="cap source sentences processed this run (pilot)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="client parallel requests (pairs with the server's "
                         "OLLAMA_NUM_PARALLEL; 1 if the server is serialized)")
    ap.add_argument("--limit-output-tokens", type=int, default=24,
                    help="Ollama num_predict cap (translations are ~10 tokens; "
                         "benchmarked free vs 48 — the rest was rambling)")
    ap.add_argument("--fewshot", action="store_true",
                    help="include the in-prompt few-shot block. Off by default: "
                         "the fine-tuned translator doesn't need it and dropping "
                         "it cuts prefill at no quality cost (chrF 67.7→68.1).")
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--repeat-last-n", type=int, default=64)
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input} — run "
              f"build_english_source.py first", file=sys.stderr)
        return 1

    sources = [ln.strip() for ln in
               args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.max_sentences:
        sources = sources[: args.max_sentences]
    print(f"Loaded {len(sources):,} English sentences from {args.input}.")

    # Resumability: .done lists every English sentence processed (even 0-yield).
    done_path = args.output.with_suffix(args.output.suffix + ".done")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if done_path.exists():
        with done_path.open(encoding="utf-8") as fh:
            done = {ln.strip() for ln in fh if ln.strip()}
    remaining = [s for s in sources if s not in done]
    print(f"Already done: {len(done):,}.  To process: {len(remaining):,}.")
    if not remaining:
        print("Nothing to do — re-running is a no-op.")
        return 0

    # Global cross-run dedup on output TP (many English sentences collapse to
    # the same simple TP; keep the corpus distinct).
    global_seen: set[str] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    global_seen.add(json.loads(raw)["text"].lower().strip())
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"Preloaded {len(global_seen):,} existing TP texts for global dedup.")

    rejects: collections.Counter = collections.Counter()
    accepted = global_dups = total_tokens = 0
    start = time.monotonic()
    out_fh = args.output.open("a", encoding="utf-8")
    done_fh = done_path.open("a", encoding="utf-8")

    def _work(eng: str):
        try:
            return _translate_one(eng, args.model, args.limit_output_tokens,
                                  args.repeat_penalty, args.repeat_last_n,
                                  args.fewshot)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return None, f"api_error:{type(exc).__name__}", 0

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        fut_to_eng = {pool.submit(_work, s): s for s in remaining}
        for fut in concurrent.futures.as_completed(fut_to_eng):
            eng = fut_to_eng[fut]
            n_done += 1
            record, reason, toks = fut.result()
            total_tokens += toks
            if record is not None:
                key = record["text"].lower().strip()
                if key in global_seen:
                    global_dups += 1
                    reason = "global_duplicate"
                else:
                    global_seen.add(key)
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    accepted += 1
            if reason:
                rejects[reason] += 1
            done_fh.write(eng + "\n")
            done_fh.flush()
            if n_done % 50 == 0 or n_done == len(remaining):
                elapsed = time.monotonic() - start
                rate = total_tokens / elapsed if elapsed > 0 else 0.0
                print(f"[{n_done}/{len(remaining)}] accepted={accepted} "
                      f"dups={global_dups} tok/s={rate:.0f}", flush=True)

    out_fh.close()
    done_fh.close()

    n = len(remaining)
    print("\n=== Translation summary ===")
    print(f"  Processed:      {n:,}")
    print(f"  Accepted:       {accepted:,} ({100*accepted/n:.1f}%)")
    print(f"  Global dups:    {global_dups:,}")
    print(f"  Tokens (gross): {total_tokens:,}")
    print("  Reject reasons:")
    for reason, c in rejects.most_common():
        print(f"    {reason:20s} {c:,}")
    print(f"  → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
