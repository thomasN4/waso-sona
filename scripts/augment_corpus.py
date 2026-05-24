"""Expand the Toki Pona corpus by paraphrasing/continuing real sentences.

Generation is anchored on real sentences from data/processed/sentences.txt,
sent to a local Ollama server (default gemma4:e2b) with few-shot examples.
Each generated sentence is filtered strictly before being written to
data/processed/synthetic.jsonl.  Re-running is a no-op (seeds already
present in the output file are skipped).

Usage::

    python scripts/augment_corpus.py --max-seeds 100
    python scripts/augment_corpus.py --model gemma4:e2b --concurrency 2
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.glyphs import WORD_TO_CODEPOINT  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "sentences.txt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "synthetic.jsonl"

TP_VOCAB: frozenset[str] = frozenset(WORD_TO_CODEPOINT)

# Stolen from bench_gemma.py — these ground the model in real TP prose.
FEWSHOT_EXAMPLES = [
    "tenpo suno ni la mi tawa ma kasi. mi lukin e waso mute. "
    "ona li kalama pona lon kasi. mi pilin pona tan ona.",
    "jan lili li lon tomo. ona li lukin e lipu. mama li toki tawa ona. "
    "tenpo lili la ona li lape.",
    "telo sewi li kama. mi lon nasin. len mi li jaki tan telo. "
    "taso mi pilin pona. ma li kama pona tan telo sewi.",
]

_WORD_RE = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _fewshot_block() -> str:
    return "\n".join(f"- {ex}" for ex in FEWSHOT_EXAMPLES)


def _paraphrase_prompt(seed: str) -> str:
    return (
        "You are an assistant who writes natural Toki Pona prose. "
        "Output only Toki Pona text — no English, no markdown, no labels.\n\n"
        f"Here are examples of natural Toki Pona:\n{_fewshot_block()}\n\n"
        "Write one paraphrase of the following Toki Pona sentence, "
        "keeping the same meaning but using different words:\n"
        f"{seed}\n\n"
        "Output only the paraphrase sentence, nothing else."
    )


def _continuation_prompt(seed: str) -> str:
    return (
        "You are an assistant who writes natural Toki Pona prose. "
        "Output only Toki Pona text — no English, no markdown, no labels.\n\n"
        f"Here are examples of natural Toki Pona:\n{_fewshot_block()}\n\n"
        "Continue the following Toki Pona sentence with 1-2 more natural sentences:\n"
        f"{seed}\n\n"
        "Output only the continuation sentences, nothing else."
    )


# ---------------------------------------------------------------------------
# Strict filter — returns (ok: bool, reason: str)
# ---------------------------------------------------------------------------

def _filter_sentence(sentence: str) -> tuple[bool, str]:
    words = _WORD_RE.findall(sentence)
    if not words:
        return False, "empty"

    # Rule 4: length (token count = number of words)
    n = len(words)
    if n < 3:
        return False, "too_short"
    if n > 40:
        return False, "too_long"

    lower_words = [w.lower() for w in words]

    # Rule 1: unknown lowercase words (zero tolerance)
    for orig, lower in zip(words, lower_words):
        if orig[0].isupper():
            continue  # proper name — allowed
        if lower not in TP_VOCAB:
            return False, "unknown_word"

    # Rule 2: double-particle errors
    text_lc = " " + sentence.lower() + " "
    if re.search(r'\bli\s+li\b', text_lc):
        return False, "double_li"
    if re.search(r'\bli\s+e\b', text_lc):
        return False, "li_e_error"
    # Leading "li" without a preceding subject (sentence starts with li)
    if re.match(r'\s*li\b', sentence.lower()):
        return False, "leading_li"

    # Rule 3: repetition
    counts: dict[str, int] = {}
    for w in lower_words:
        counts[w] = counts.get(w, 0) + 1
    if max(counts.values()) / n > 0.4:
        return False, "unigram_repetition"

    bigrams = [(lower_words[i], lower_words[i + 1]) for i in range(n - 1)]
    if bigrams:
        bg_counts: dict[tuple[str, str], int] = {}
        for bg in bigrams:
            bg_counts[bg] = bg_counts.get(bg, 0) + 1
        if max(bg_counts.values()) > 2:
            return False, "bigram_repetition"

    # Rule 5: missing predicate
    has_li = "li" in lower_words
    starts_with_subject = lower_words[0] in {"mi", "sina", "o"}
    if not has_li and not starts_with_subject:
        return False, "missing_predicate"

    return True, ""


# ---------------------------------------------------------------------------
# Ollama HTTP helper
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
    """POST to Ollama /api/generate. Returns (response_text, eval_count)."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    text = data.get("response", "").strip()
    eval_count = data.get("eval_count", 0)
    return text, eval_count


def _split_into_sentences(text: str) -> list[str]:
    """Split a multi-sentence response into individual sentences."""
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Per-seed processing
# ---------------------------------------------------------------------------

def _process_seed(
    seed: str,
    model: str,
    paraphrase_n: int,
    continuation_n: int,
    max_tokens: int,
) -> tuple[list[dict], collections.Counter, int]:
    """Generate, filter, and return records for one seed.

    Returns (accepted_records, reject_counts, total_tokens_generated).
    """
    accepted: list[dict] = []
    rejects: collections.Counter = collections.Counter()
    total_tokens = 0

    # --- paraphrase attempts ---
    for _ in range(paraphrase_n):
        try:
            text, toks = _ollama_generate(_paraphrase_prompt(seed), model, max_tokens)
            total_tokens += toks
            # Ask for one sentence; take first sentence from response.
            for sent in _split_into_sentences(text)[:1]:
                ok, reason = _filter_sentence(sent)
                if ok:
                    accepted.append({
                        "source": f"{model}/paraphrase",
                        "seed": seed,
                        "mode": "paraphrase",
                        "text": sent,
                    })
                else:
                    rejects[reason] += 1
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            rejects["api_error"] += 1
            print(f"    [api_error paraphrase] {exc}", file=sys.stderr)

    # --- continuation attempts ---
    for _ in range(continuation_n):
        try:
            text, toks = _ollama_generate(_continuation_prompt(seed), model, max_tokens)
            total_tokens += toks
            # Up to 2 sentences from the continuation response.
            for sent in _split_into_sentences(text)[:2]:
                ok, reason = _filter_sentence(sent)
                if ok:
                    accepted.append({
                        "source": f"{model}/continuation",
                        "seed": seed,
                        "mode": "continuation",
                        "text": sent,
                    })
                else:
                    rejects[reason] += 1
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            rejects["api_error"] += 1
            print(f"    [api_error continuation] {exc}", file=sys.stderr)

    return accepted, rejects, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment Toki Pona corpus via Ollama/Gemma paraphrase+continuation."
    )
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--paraphrase-n", type=int, default=5,
                        help="Paraphrase attempts per seed (default 5)")
    parser.add_argument("--continuation-n", type=int, default=3,
                        help="Continuation attempts per seed (default 3)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="ThreadPoolExecutor max_workers (default 2)")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Limit number of seeds processed (for testing)")
    parser.add_argument("--limit-output-tokens", type=int, default=128,
                        help="Ollama num_predict cap per call (default 128)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to sentences.txt")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Path to synthetic.jsonl (appended)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load seeds
    all_seeds = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    if args.max_seeds:
        all_seeds = all_seeds[: args.max_seeds]
    print(f"Loaded {len(all_seeds)} seed sentences from {input_path}.")

    # Resumability: collect seeds already in output
    done_seeds: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                    done_seeds.add(rec["seed"])
                except (json.JSONDecodeError, KeyError):
                    pass
    remaining = [s for s in all_seeds if s not in done_seeds]
    print(f"Already done: {len(done_seeds)} seeds.  To process: {len(remaining)}.")

    if not remaining:
        print("Nothing to do — re-running is a no-op.")
        return

    # Aggregate stats (updated from each future's result in the main thread)
    global_rejects: collections.Counter = collections.Counter()
    total_accepted = 0
    total_candidates = 0
    total_tokens = 0
    start = time.monotonic()

    out_fh = output_path.open("a", encoding="utf-8")

    def _work(seed: str) -> tuple[list[dict], collections.Counter, int]:
        return _process_seed(
            seed,
            model=args.model,
            paraphrase_n=args.paraphrase_n,
            continuation_n=args.continuation_n,
            max_tokens=args.limit_output_tokens,
        )

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_seed = {pool.submit(_work, s): s for s in remaining}
        for fut in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[fut]
            n_done += 1
            try:
                records, rejects, toks = fut.result()
            except Exception as exc:
                print(f"  FATAL error for seed {seed!r}: {exc}", file=sys.stderr)
                records, rejects, toks = [], collections.Counter({"exception": 1}), 0

            for rec in records:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

            n_cands = sum(rejects.values()) + len(records)
            total_accepted += len(records)
            total_candidates += n_cands
            total_tokens += toks
            global_rejects.update(rejects)

            elapsed = time.monotonic() - start
            rate = total_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"[{n_done}/{len(remaining)}] "
                f"seed_accepted={len(records)} cands={n_cands} "
                f"tok/s={rate:.1f}  {seed[:70]!r}"
            )

    out_fh.close()

    elapsed = time.monotonic() - start
    rate = total_tokens / elapsed if elapsed > 0 else 0.0

    print("\n=== Augmentation summary ===")
    print(f"  Seeds processed:      {len(remaining)}")
    print(f"  Candidates generated: {total_candidates}")
    print(f"  Accepted:             {total_accepted}")
    pct = total_accepted / max(total_candidates, 1) * 100
    print(f"  Accept rate:          {pct:.1f}%")
    print(f"  Total tokens:         {total_tokens}")
    print(f"  Wall time:            {elapsed:.1f}s")
    print(f"  Effective tok/s:      {rate:.1f}")
    print(f"\n  Reject-reason histogram (most common first):")
    for reason, count in global_rejects.most_common():
        bar = "#" * min(count, 40)
        print(f"    {reason:25s} {count:6d}  {bar}")
    print(f"\n  Output: {output_path}")


if __name__ == "__main__":
    main()
