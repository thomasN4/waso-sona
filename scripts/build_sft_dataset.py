"""Build an instruction-tuning dataset for QLoRA fine-tuning of Gemma 4.

Reads the real Toki Pona corpus from data/processed/corpus.filtered.jsonl and
emits two HF messages-format JSONL files (sft_train.jsonl + sft_val.jsonl)
whose user-side prompts match the exact shape that scripts/augment_corpus.py
sends at inference time.

Two example types:

* Continuation pairs (main signal) — for multi-sentence chunks: user message
  is `_continuation_prompt(prefix)` from augment_corpus.py (few-shot block
  included verbatim); model response is the next 1-2 sentences.
* Topic prompts (diversity) — rotates ~10 instruction templates against a
  random chunk as the response.

Each model response is passed through `augment_corpus._filter_sentence` (the
strict filter applied at inference) so the SFT signal matches the bar the
model will be judged by. Responses are deduped by lowercase-normalized hash,
and every example is rendered through the Gemma 4 chat template to enforce
a `--max-tokens` cap (default 1024).

Usage::

    python scripts/build_sft_dataset.py --limit 200  # smoke test
    python scripts/build_sft_dataset.py              # full run
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

import augment_corpus  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "corpus.filtered.jsonl"
DEFAULT_OUT_TRAIN = REPO_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_OUT_VAL = REPO_ROOT / "data" / "processed" / "sft_val.jsonl"
DEFAULT_PARAPHRASE_PAIRS = REPO_ROOT / "data" / "processed" / "paraphrase_pairs.jsonl"
DEFAULT_TRANSLATION_PAIRS = REPO_ROOT / "data" / "processed" / "translation_pairs.jsonl"
DEFAULT_OUT_TRANSLATE_TRAIN = REPO_ROOT / "data" / "processed" / "sft_translate_train.jsonl"
DEFAULT_OUT_TRANSLATE_VAL = REPO_ROOT / "data" / "processed" / "sft_translate_val.jsonl"
MODEL_ID = "google/gemma-4-E2B-it"

# Copied from scripts/fetch_data.py:201 — sentence-final punctuation OR newline.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

MAX_CHUNK_CHARS = 600

TOPIC_PROMPTS = [
    "Write a few sentences in Toki Pona about everyday life.",
    "Compose a short Toki Pona reflection.",
    "Write a brief Toki Pona description of a scene.",
    "Write a short Toki Pona story (3-5 sentences).",
    "Write a few sentences of Toki Pona prose.",
    "Compose a short Toki Pona observation about nature.",
    "Write a Toki Pona diary entry, a few sentences long.",
    "Write a Toki Pona thought, no more than 3 sentences.",
    "Compose a few natural sentences in Toki Pona.",
    "Write a brief Toki Pona narrative.",
]


# ---------------------------------------------------------------------------
# Chunking and pair construction
# ---------------------------------------------------------------------------

def _split_sents(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _ensure_terminal(s: str) -> str:
    return s if s.endswith((".", "!", "?")) else s + "."


def chunk_doc(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a doc into ≤ max_chars chunks at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sents = _split_sents(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for s in sents:
        s = _ensure_terminal(s)
        if buf_len + len(s) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [s], len(s)
        else:
            buf.append(s)
            buf_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


MIN_PREFIX_WORDS = 5  # enough context for a meaningful continuation


def make_continuation_pair(
    chunk: str,
    rng: random.Random,
    max_prefix_sents: int = 3,
    max_suffix_sents: int = 2,
) -> tuple[str, str] | None:
    """Split chunk into (prefix, suffix). None if chunk has < 2 sentences or
    the prefix is too short to be a meaningful continuation seed (e.g. the
    bulk of single-sentence sources like Tatoeba)."""
    sents = _split_sents(chunk)
    if len(sents) < 2:
        return None
    split_at = rng.randint(1, min(max_prefix_sents, len(sents) - 1))
    prefix = " ".join(_ensure_terminal(s) for s in sents[:split_at])
    if len(augment_corpus._WORD_RE.findall(prefix)) < MIN_PREFIX_WORDS:
        return None
    suffix = " ".join(
        _ensure_terminal(s) for s in sents[split_at:split_at + max_suffix_sents]
    )
    return prefix, suffix


# ---------------------------------------------------------------------------
# Filter (per-sentence, strict — same bar augment_corpus uses at inference)
# ---------------------------------------------------------------------------

def filter_response(text: str) -> tuple[bool, str]:
    sents = _split_sents(text)
    if not sents:
        return False, "empty"
    for s in sents:
        ok, reason = augment_corpus._filter_sentence(s)
        if not ok:
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Dedup / tokenization helpers
# ---------------------------------------------------------------------------

def dedup_key(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def render_token_count(tok, messages: list[dict]) -> int:
    # transformers 5 returns a BatchEncoding (len == number of fields); take
    # input_ids explicitly so we measure token count.
    enc = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=False)
    return len(enc["input_ids"])


def percentiles(xs: list[int], ps=(0.5, 0.9, 0.99)) -> dict[str, int]:
    if not xs:
        return {f"p{int(p*100)}": 0 for p in ps} | {"max": 0}
    s = sorted(xs)
    n = len(s)
    out = {f"p{int(p*100)}": s[min(int(p * n), n - 1)] for p in ps}
    out["max"] = s[-1]
    return out


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------

def build_corpus_examples(
    args, rng: random.Random,
) -> tuple[list[dict], collections.Counter]:
    """The multi-task teacher dataset: continuation + topic + paraphrase from
    the real corpus. (Unchanged from the original inline pipeline.)"""
    print(f"Reading {args.input}…", flush=True)
    docs: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    if args.limit:
        docs = docs[: args.limit]
    print(f"  {len(docs)} docs", flush=True)

    # ---- source balance ----
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)
    if "tatoeba" in by_source and len(by_source["tatoeba"]) > args.max_tatoeba:
        rng.shuffle(by_source["tatoeba"])
        by_source["tatoeba"] = by_source["tatoeba"][: args.max_tatoeba]
        print(f"  capped tatoeba → {len(by_source['tatoeba'])}", flush=True)
    selected = [d for src_docs in by_source.values() for d in src_docs]
    rng.shuffle(selected)
    print(f"  after balancing: {len(selected)} docs", flush=True)

    # ---- chunk + generate examples ----
    examples: list[dict] = []
    chunks_by_source: collections.Counter = collections.Counter()
    for d in selected:
        chunks = chunk_doc(d["text"])
        chunks_by_source[d["source"]] += len(chunks)
        for chunk in chunks:
            # Topic-prompt branch: probability args.topic_prompt_frac per chunk.
            if rng.random() < args.topic_prompt_frac:
                template = rng.choice(TOPIC_PROMPTS)
                examples.append({
                    "source": d["source"],
                    "id": d["id"],
                    "mode": "topic",
                    "messages": [
                        {"role": "user", "content": template},
                        {"role": "assistant", "content": chunk},
                    ],
                })
                continue
            # Continuation branch: only works if chunk has ≥ 2 sentences.
            pair = make_continuation_pair(chunk, rng)
            if pair is None:
                continue
            prefix, suffix = pair
            examples.append({
                "source": d["source"],
                "id": d["id"],
                "mode": "continuation",
                "messages": [
                    {"role": "user",
                     "content": augment_corpus._continuation_prompt(prefix)},
                    {"role": "assistant", "content": suffix},
                ],
            })
    # ---- paraphrase examples (3rd mode) ----
    # TP paraphrase pairs (Tatoeba clusters, build_paraphrase_pairs.py). The
    # user message is the exact inference shape (_paraphrase_prompt); the
    # response is the sibling sentence. Added directly (bypass doc balancing).
    if args.paraphrase_pairs.exists():
        pairs = [json.loads(ln) for ln in
                 args.paraphrase_pairs.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng.shuffle(pairs)
        if args.max_paraphrase:
            pairs = pairs[: args.max_paraphrase]
        for p in pairs:
            examples.append({
                "source": "paraphrase",
                "id": str(p.get("eng_id", "")),
                "mode": "paraphrase",
                "messages": [
                    {"role": "user",
                     "content": augment_corpus._paraphrase_prompt(p["a"])},
                    {"role": "assistant", "content": p["b"]},
                ],
            })
        print(f"  added {len(pairs)} paraphrase pairs", flush=True)
    else:
        print(f"  (no paraphrase pairs at {args.paraphrase_pairs}; skipping)",
              flush=True)
    return examples, chunks_by_source


def build_translation_examples(
    pairs_path: Path, rng: random.Random,
) -> tuple[list[dict], collections.Counter]:
    """The standalone translator dataset: one en→tp example per parallel pair
    from build_translation_pairs.py. User message is the exact inference shape
    (`_translate_prompt`); response is the TP target. `id` is the eng_id so the
    downstream hash split stays by English sentence."""
    if not pairs_path.exists():
        print(f"ERROR: translation pairs not found: {pairs_path}", file=sys.stderr)
        raise SystemExit(1)
    pairs = [json.loads(ln) for ln in
             pairs_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng.shuffle(pairs)
    examples: list[dict] = []
    for p in pairs:
        examples.append({
            "source": p.get("source", "tatoeba"),
            "id": str(p.get("eng_id", "")),
            "mode": "translation",
            "messages": [
                {"role": "user",
                 "content": augment_corpus._translate_prompt(p["en"])},
                {"role": "assistant", "content": p["tp"]},
            ],
        })
    print(f"  translation pairs: {len(examples)}", flush=True)
    return examples, collections.Counter()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-train", type=Path, default=DEFAULT_OUT_TRAIN)
    ap.add_argument("--output-val", type=Path, default=DEFAULT_OUT_VAL)
    ap.add_argument("--model-id", default=MODEL_ID,
                    help="HF model id used only for its tokenizer + chat template")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tatoeba", type=int, default=20000,
                    help="cap tatoeba random sample (default 20000)")
    ap.add_argument("--paraphrase-pairs", type=Path, default=DEFAULT_PARAPHRASE_PAIRS,
                    help="JSONL of {a,b} TP paraphrase pairs (build_paraphrase_pairs.py)")
    ap.add_argument("--max-paraphrase", type=int, default=None,
                    help="cap paraphrase examples added (default: all)")
    ap.add_argument("--topic-prompt-frac", type=float, default=0.20,
                    help="probability a chunk becomes a topic-prompt example (default 0.20)")
    ap.add_argument("--translation-only", action="store_true",
                    help="build a standalone en→tp translator SFT from "
                         "--translation-pairs (skips the corpus/teacher modes)")
    ap.add_argument("--translation-pairs", type=Path,
                    default=DEFAULT_TRANSLATION_PAIRS,
                    help="JSONL of {en,tp,eng_id} pairs (build_translation_pairs.py)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap docs read from input (for smoke testing)")
    args = ap.parse_args(argv)

    # In translation-only mode, default the outputs to the translator files
    # unless the caller overrode them explicitly.
    if args.translation_only:
        if args.output_train == DEFAULT_OUT_TRAIN:
            args.output_train = DEFAULT_OUT_TRANSLATE_TRAIN
        if args.output_val == DEFAULT_OUT_VAL:
            args.output_val = DEFAULT_OUT_TRANSLATE_VAL

    rng = random.Random(args.seed)

    print(f"Loading tokenizer {args.model_id}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)

    if args.translation_only:
        examples, chunks_by_source = build_translation_examples(
            args.translation_pairs, rng)
    else:
        if not args.input.exists():
            print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
            return 1
        examples, chunks_by_source = build_corpus_examples(args, rng)

    print(f"\nRaw examples: {len(examples)}", flush=True)
    print(f"  chunks per source: {dict(chunks_by_source)}", flush=True)

    # ---- filter responses (strict, matches augment_corpus inference filter) ----
    print("\nFiltering responses…", flush=True)
    filter_rejects: collections.Counter = collections.Counter()
    keep_by_source: collections.Counter = collections.Counter()
    seen_by_source: collections.Counter = collections.Counter()
    filtered: list[dict] = []
    for ex in examples:
        seen_by_source[ex["source"]] += 1
        ok, reason = filter_response(ex["messages"][1]["content"])
        if ok:
            filtered.append(ex)
            keep_by_source[ex["source"]] += 1
        else:
            filter_rejects[reason] += 1
    pct = len(filtered) / max(len(examples), 1) * 100
    print(f"  passed: {len(filtered)}/{len(examples)} ({pct:.1f}%)", flush=True)
    for reason, count in filter_rejects.most_common():
        print(f"    {reason:25s} {count}", flush=True)
    print("  keep-rate by source:", flush=True)
    for src in sorted(seen_by_source):
        s, k = seen_by_source[src], keep_by_source[src]
        print(f"    {src:10s} {k:>6}/{s:<6} ({k/s*100 if s else 0:5.1f}%)",
              flush=True)

    # ---- dedup on response ----
    print("\nDeduping by response hash…", flush=True)
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in filtered:
        # Mode-aware: paraphrase clusters reuse the same target across
        # different sources, so dedup on (prompt, response) to keep distinct
        # a→b pairs. Continuation/topic stay response-only (unchanged).
        if ex["mode"] in ("paraphrase", "translation"):
            key = dedup_key(ex["messages"][0]["content"] + "␟"
                            + ex["messages"][1]["content"])
        else:
            key = dedup_key(ex["messages"][1]["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    pct = len(deduped) / max(len(filtered), 1) * 100
    print(f"  unique: {len(deduped)}/{len(filtered)} ({pct:.1f}%)", flush=True)

    # ---- token-count check ----
    print(f"\nRendering chat template, enforcing ≤ {args.max_tokens} tokens…",
          flush=True)
    within: list[dict] = []
    tok_counts: list[int] = []
    for ex in deduped:
        n = render_token_count(tok, ex["messages"])
        if n <= args.max_tokens:
            within.append(ex)
            tok_counts.append(n)
    print(f"  within budget: {len(within)}/{len(deduped)}", flush=True)
    if tok_counts:
        p = percentiles(tok_counts)
        print(f"  token percentiles: p50={p['p50']}, p90={p['p90']}, "
              f"p99={p['p99']}, max={p['max']}", flush=True)

    mode_counts = collections.Counter(ex["mode"] for ex in within)
    final_src = collections.Counter(ex["source"] for ex in within)
    print(f"\n  final by mode:   {dict(mode_counts)}", flush=True)
    print(f"  final by source: {dict(final_src)}", flush=True)

    # ---- deterministic train/val split by id hash ----
    cutoff = int(args.val_frac * 1000)
    train: list[dict] = []
    val: list[dict] = []
    for ex in within:
        h = int(hashlib.sha256(ex["id"].encode()).hexdigest(), 16) % 1000
        (val if h < cutoff else train).append(ex)
    print(f"\nSplit: {len(train)} train, {len(val)} val "
          f"(val_frac={args.val_frac})", flush=True)

    # ---- write JSONL ----
    for path, recs in [(args.output_train, train), (args.output_val, val)]:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"  wrote {shown}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
