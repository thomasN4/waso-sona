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

from sitelen.glyphs import WORD_TO_CODEPOINT  # noqa: E402

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "waso-teacher"
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

# Task-demonstrating few-shot for paraphrasing: real Tatoeba sibling pairs
# (same English translation) showing structural rewrites — la-fronting↔tawa,
# la-time↔lon-time, negated-question↔anu-question. The model paraphrases far
# more faithfully when the prompt shows the *task* (A→B) rather than generic
# prose; this list is the in-prompt demonstration and matches the SFT prompt.
PARAPHRASE_FEWSHOT = [
    ("mi la sitelen esun li ike.", "sitelen esun li ike tawa mi."),
    ("tenpo ni la mi wile tawa weka.", "mi wile weka lon tenpo ni."),
    ("mi ken ala ken open e lupa lili?", "mi ken open e lupa lili anu seme?"),
]

# Task-demonstrating few-shot for English→Toki Pona translation (real, short
# pairs from data/processed/translation_pairs.jsonl). Shapes the model toward
# concise faithful renderings rather than verbose drift. Matches the SFT prompt
# built by build_sft_dataset.py --translation-only, so train == inference.
TRANSLATE_FEWSHOT = [
    ("This is a person.", "ni li jan."),
    ("Go to bed!", "o tawa supa lape!"),
    ("It's no big deal.", "ni li ijo suli ala."),
]

_WORD_RE = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _fewshot_block() -> str:
    return "\n".join(f"- {ex}" for ex in FEWSHOT_EXAMPLES)


def _paraphrase_fewshot_block() -> str:
    return "\n".join(f"- {a} → {b}" for a, b in PARAPHRASE_FEWSHOT)


def _translate_fewshot_block() -> str:
    return "\n".join(f"- {en} → {tp}" for en, tp in TRANSLATE_FEWSHOT)


def _paraphrase_prompt(seed: str) -> str:
    return (
        "You are an assistant who paraphrases Toki Pona. "
        "Output only Toki Pona text — no English, no markdown, no labels.\n\n"
        "Here are examples of paraphrasing — same meaning, different words "
        f"(original → paraphrase):\n{_paraphrase_fewshot_block()}\n\n"
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


def _translate_prompt(english: str, fewshot: bool = True) -> str:
    examples = (
        "Here are examples of English → Toki Pona translation:\n"
        f"{_translate_fewshot_block()}\n\n"
    ) if fewshot else ""
    return (
        "You are an assistant who translates English into Toki Pona. "
        "Output only Toki Pona text — no English, no markdown, no labels.\n\n"
        f"{examples}"
        "Translate the following English sentence into Toki Pona:\n"
        f"{english}\n\n"
        "Output only the Toki Pona translation, nothing else."
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

    # Rule 5: missing predicate. A valid TP sentence has `li`, OR contains
    # one of the three subjects that drop `li` (`mi`/`sina`/`o`) anywhere —
    # they may follow a `la`-clause, e.g. "tenpo suno ni la mi tawa ma kasi".
    has_li = "li" in lower_words
    has_li_dropping_subject = any(w in {"mi", "sina", "o"} for w in lower_words)
    if not has_li and not has_li_dropping_subject:
        return False, "missing_predicate"

    return True, ""


# ---------------------------------------------------------------------------
# Ollama HTTP helper
# ---------------------------------------------------------------------------

def _ollama_generate(
    prompt: str, model: str, max_tokens: int,
    repeat_penalty: float, repeat_last_n: int,
) -> tuple[str, int]:
    """POST to Ollama /api/chat. Returns (response_text, eval_count).

    Uses /api/chat (not /api/generate) so the fine-tuned teacher's chat
    template (carried into the GGUF) wraps the prompt the same way it was
    trained — `_continuation_prompt`/`_paraphrase_prompt` are the user turn.
    A raw /api/generate prompt would skip the `<|turn>` markers and put the
    model off its training distribution.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        # temp=0.8 is the real repetition fix (greedy loops badly); the
        # repeat_* knobs are exposed but had negligible effect at this temp
        # on TP (its tiny vocab forces grammatical function-word reuse).
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.8,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
        },
    }).encode()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    text = data.get("message", {}).get("content", "").strip()
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
    repeat_penalty: float,
    repeat_last_n: int,
) -> tuple[list[dict], collections.Counter, int]:
    """Generate, filter, and return records for one seed.

    Returns (accepted_records, reject_counts, total_tokens_generated).
    """
    accepted: list[dict] = []
    rejects: collections.Counter = collections.Counter()
    total_tokens = 0
    # Per-seed dedup: Gemma reliably re-emits the same paraphrase across
    # attempts. Key by lowercased text so casing differences don't slip
    # through as "new" records.
    seen: set[str] = set()

    def _consider(sent: str, mode: str, take_limit: int) -> None:
        ok, reason = _filter_sentence(sent)
        if not ok:
            rejects[reason] += 1
            return
        key = sent.lower().strip()
        if key in seen:
            rejects["duplicate"] += 1
            return
        seen.add(key)
        accepted.append({
            "source": f"{model}/{mode}",
            "seed": seed,
            "mode": mode,
            "text": sent,
        })

    # --- paraphrase attempts ---
    for _ in range(paraphrase_n):
        try:
            text, toks = _ollama_generate(_paraphrase_prompt(seed), model, max_tokens,
                                          repeat_penalty, repeat_last_n)
            total_tokens += toks
            for sent in _split_into_sentences(text)[:1]:
                _consider(sent, "paraphrase", 1)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            rejects["api_error"] += 1
            print(f"    [api_error paraphrase] {exc}", file=sys.stderr)

    # --- continuation attempts ---
    for _ in range(continuation_n):
        try:
            text, toks = _ollama_generate(_continuation_prompt(seed), model, max_tokens,
                                          repeat_penalty, repeat_last_n)
            total_tokens += toks
            for sent in _split_into_sentences(text)[:2]:
                _consider(sent, "continuation", 2)
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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model — the fine-tuned teacher (default {DEFAULT_MODEL})")
    parser.add_argument("--paraphrase-n", type=int, default=5,
                        help="Paraphrase attempts per seed (default 5)")
    parser.add_argument("--continuation-n", type=int, default=3,
                        help="Continuation attempts per seed (default 3)")
    parser.add_argument("--repeat-penalty", type=float, default=1.1,
                        help="Ollama repeat_penalty (default 1.1; tuning showed "
                             "negligible effect on TP at temp 0.8)")
    parser.add_argument("--repeat-last-n", type=int, default=64,
                        help="Ollama repeat_last_n window (default 64)")
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

    # Resumability: a sidecar file lists every seed that has been processed,
    # even if it produced zero acceptances. Without this, low-yield seeds
    # would be retried on every run and the script could never reach a
    # no-op steady state.
    done_path = output_path.with_suffix(output_path.suffix + ".done")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_seeds: set[str] = set()
    if done_path.exists():
        with done_path.open(encoding="utf-8") as fh:
            done_seeds = {ln.strip() for ln in fh if ln.strip()}
    elif output_path.exists():
        # Migration: derive done set from any existing output records.
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

    # Global cross-seed / cross-run dedup: the teacher re-emits the same
    # sentence across different seeds, and re-runs append. Preload every text
    # already in the output so the student corpus stays duplicate-free.
    global_seen: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    global_seen.add(json.loads(raw)["text"].lower().strip())
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"Preloaded {len(global_seen)} existing texts for global dedup.")

    # Aggregate stats (updated from each future's result in the main thread)
    global_rejects: collections.Counter = collections.Counter()
    total_accepted = 0
    total_candidates = 0
    total_tokens = 0
    global_dups = 0
    start = time.monotonic()

    out_fh = output_path.open("a", encoding="utf-8")
    done_fh = done_path.open("a", encoding="utf-8")

    def _work(seed: str) -> tuple[list[dict], collections.Counter, int]:
        return _process_seed(
            seed,
            model=args.model,
            paraphrase_n=args.paraphrase_n,
            continuation_n=args.continuation_n,
            max_tokens=args.limit_output_tokens,
            repeat_penalty=args.repeat_penalty,
            repeat_last_n=args.repeat_last_n,
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

            kept = 0
            for rec in records:
                key = rec["text"].lower().strip()
                if key in global_seen:
                    global_dups += 1
                    continue
                global_seen.add(key)
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            out_fh.flush()
            done_fh.write(seed + "\n")
            done_fh.flush()

            n_cands = sum(rejects.values()) + len(records)
            total_accepted += kept
            total_candidates += n_cands
            total_tokens += toks
            global_rejects.update(rejects)

            elapsed = time.monotonic() - start
            rate = total_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"[{n_done}/{len(remaining)}] "
                f"kept={kept} cands={n_cands} "
                f"tok/s={rate:.1f}  {seed[:70]!r}"
            )

    out_fh.close()
    done_fh.close()

    elapsed = time.monotonic() - start
    rate = total_tokens / elapsed if elapsed > 0 else 0.0

    print("\n=== Augmentation summary ===")
    print(f"  Seeds processed:      {len(remaining)}")
    print(f"  Candidates generated: {total_candidates}")
    print(f"  Accepted (written):   {total_accepted}")
    print(f"  Global dups dropped:  {global_dups}")
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
