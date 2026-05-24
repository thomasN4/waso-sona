"""Benchmark Gemma 4 variants for Toki Pona generation.

Two passes:

1. **Quality** — N prompts per model with diverse topics/styles, saved as
   JSONL for eyeballing. Each record carries the prompt, the model output,
   and Ollama's own token-count + duration fields.

2. **Throughput** — timed bursts at concurrency 1/2/4 per model on a fixed
   short prompt, reporting effective tok/s and the wall-clock projection to
   100M / 1B tokens.

The Ollama server must be running locally (default http://localhost:11434).
Pure stdlib; no third-party deps.

Usage::

    python scripts/bench_gemma.py
    python scripts/bench_gemma.py --models gemma4:e2b --quality-n 20
    python scripts/bench_gemma.py --skip-quality   # only throughput
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.glyphs import WORD_TO_CODEPOINT  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
BENCH_DIR = REPO_ROOT / "data/bench"

MODELS = ["gemma4:e2b", "gemma4-e4b-iq4xs:latest", "gemma4:latest"]

TP_VOCAB = set(WORD_TO_CODEPOINT)
TP_LETTERS = set("aeijklmnopstuw")

# Heavy system messages collapse small Gemma variants into nonsense; relying
# on few-shot exemplars in the user prompt works much better.
SYSTEM = (
    "You are an assistant who writes natural Toki Pona prose. "
    "Output only Toki Pona text — no English, no markdown, no labels."
)

# Three known-good seed examples shown to the model with every prompt.
FEWSHOT_EXAMPLES = [
    "tenpo suno ni la mi tawa ma kasi. mi lukin e waso mute. "
    "ona li kalama pona lon kasi. mi pilin pona tan ona.",
    "jan lili li lon tomo. ona li lukin e lipu. mama li toki tawa ona. "
    "tenpo lili la ona li lape.",
    "telo sewi li kama. mi lon nasin. len mi li jaki tan telo. "
    "taso mi pilin pona. ma li kama pona tan telo sewi.",
]

TOPICS = [
    "an evening walk in the rain",
    "a child asking about the stars",
    "two friends sharing food",
    "a cat watching birds outside",
    "a memory from before sleep",
    "the feeling of warm sun on skin",
    "learning to swim in a river",
    "a quiet room with one window",
    "the smell of bread in the morning",
    "an old person remembering their youth",
    "a small boat on a big lake",
    "the sound of wind in tall grass",
    "a dream about flying",
    "the first day in a new house",
    "a market on a busy day",
    "watching the sea at night",
    "a long letter never sent",
    "a song heard once and never again",
    "the taste of cold water after running",
    "two strangers meeting on a road",
    "the moon behind clouds",
    "a child's first word",
    "a garden after the rain",
    "an empty street at dawn",
    "a fire in the cold",
]

STYLES = [
    "a short paragraph (4–6 sentences)",
    "a small story",
    "a brief diary entry",
    "a description, no characters",
    "a quiet reflection",
]

SEEDS = [
    "mi lon tomo mi. tenpo pimeja li kama.",
    "jan lili li lukin e waso lon sewi.",
    "tenpo suno ni la mi pilin pona.",
    "ona li toki e ni: ",
    "soweli mi li moku e telo. ona li ",
]


def _fewshot_block() -> str:
    return "Here are examples of natural Toki Pona prose:\n\n" + "\n\n".join(
        f"Example {i+1}: {ex}" for i, ex in enumerate(FEWSHOT_EXAMPLES)
    )


def build_prompts(n: int) -> list[str]:
    """Mix topic-prompts and seed-continuation prompts, each few-shot wrapped."""
    fewshot = _fewshot_block()
    out: list[str] = []
    for ti, topic in enumerate(TOPICS):
        style = STYLES[ti % len(STYLES)]
        out.append(
            f"{fewshot}\n\n"
            f"Now write a new Toki Pona text ({style}) about: {topic}. "
            f"Output only the Toki Pona text, no English."
        )
    for seed in SEEDS:
        out.append(
            f"{fewshot}\n\n"
            f"Continue this Toki Pona text naturally for several more "
            f"sentences. Output only Toki Pona.\n\n{seed}"
        )
    return out[:n]


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------

def call_ollama(
    model: str,
    prompt: str,
    num_predict: int = 256,
    temperature: float = 0.8,
    timeout: float = 300.0,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        # Gemma 4 has reasoning enabled by default; without this the model
        # exhausts num_predict on hidden thinking and emits an empty response.
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": 0.95,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        return {
            "response": "",
            "eval_count": 0,
            "eval_duration_ns": 0,
            "prompt_eval_count": 0,
            "wall_seconds": time.monotonic() - t0,
            "error": str(e),
        }
    return {
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "wall_seconds": time.monotonic() - t0,
    }


def warm(model: str) -> None:
    print(f"[{model}] warming…", flush=True)
    r = call_ollama(model, "toki", num_predict=4, timeout=600)
    if "error" in r:
        print(f"  ! warm failed: {r['error']}", flush=True)


# ---------------------------------------------------------------------------
# Quality pass
# ---------------------------------------------------------------------------

def quality_bench(model: str, prompts: list[str], num_predict: int) -> list[dict]:
    out: list[dict] = []
    for i, p in enumerate(prompts):
        r = call_ollama(model, p, num_predict=num_predict)
        rec = {"prompt_id": i, "prompt": p, **r}
        out.append(rec)
        tok_s = (
            r["eval_count"] / max(r["eval_duration_ns"], 1) * 1e9
            if r["eval_count"]
            else 0.0
        )
        print(
            f"  [{model} {i+1:>3}/{len(prompts)}] "
            f"{r['eval_count']:>4} tok, {r['wall_seconds']:>5.1f}s wall, "
            f"{tok_s:>5.1f} tok/s",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Throughput pass
# ---------------------------------------------------------------------------

THROUGHPUT_PROMPT = "o pali e lipu lili pona pi tenpo suno ni."


def throughput_bench(
    model: str,
    concurrency: int,
    n_requests: int,
    num_predict: int,
) -> dict:
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(call_ollama, model, THROUGHPUT_PROMPT, num_predict)
            for _ in range(n_requests)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    wall = time.monotonic() - t0
    tokens = sum(r["eval_count"] for r in results)
    errors = sum(1 for r in results if "error" in r)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "tokens": tokens,
        "wall_seconds": wall,
        "tok_per_s": tokens / wall if wall > 0 else 0.0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Filter pass (mirrors filter_corpus.py + fetch_data._looks_like_toki_pona)
# ---------------------------------------------------------------------------

import re

_TP_LINE_RE = re.compile(r"^[A-Za-z\s.,;:!?'\"-]+$")
_WORD_RE = re.compile(r"[A-Za-z]+")
TP_WORD_RATIO_THRESHOLD = 0.6


def non_tp_letter_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if c not in TP_LETTERS)
    return bad / len(letters)


def looks_like_toki_pona_sentence(sentence: str) -> bool:
    if not _TP_LINE_RE.match(sentence):
        return False
    words = _WORD_RE.findall(sentence)
    lowercase = [w for w in words if w[0].islower()]
    if not lowercase:
        return False
    hits = sum(1 for w in lowercase if w.lower() in TP_VOCAB)
    return hits / len(lowercase) >= TP_WORD_RATIO_THRESHOLD


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def analyze_quality_file(path: Path) -> dict:
    n_docs = 0
    docs_kept = 0  # passes non_tp_letter_ratio <= 0.05
    total_sents = 0
    sents_kept = 0
    total_tokens_emitted = 0
    total_chars = 0
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("response", "")
            if not text.strip():
                continue
            n_docs += 1
            total_tokens_emitted += rec.get("eval_count", 0)
            total_chars += len(text)
            if non_tp_letter_ratio(text) <= 0.05:
                docs_kept += 1
            for chunk in _SENT_SPLIT_RE.split(text):
                s = chunk.strip()
                if not s:
                    continue
                total_sents += 1
                if looks_like_toki_pona_sentence(s):
                    sents_kept += 1
    return {
        "n_docs": n_docs,
        "doc_keep_rate": docs_kept / n_docs if n_docs else 0.0,
        "n_sents": total_sents,
        "sent_keep_rate": sents_kept / total_sents if total_sents else 0.0,
        "total_tokens_emitted": total_tokens_emitted,
        "total_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--quality-n", type=int, default=30,
                    help="prompts per model in the quality pass (default: 30)")
    ap.add_argument("--num-predict", type=int, default=256,
                    help="max output tokens per call (default: 256)")
    ap.add_argument("--throughput-n", type=int, default=8,
                    help="requests per concurrency level (default: 8)")
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true")
    args = ap.parse_args(argv)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.quality_n)
    print(f"prompts: {len(prompts)} | models: {args.models}", flush=True)

    throughput: dict[str, dict] = {}
    quality_summary: dict[str, dict] = {}

    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        warm(model)

        if not args.skip_quality:
            results = quality_bench(model, prompts, args.num_predict)
            qpath = BENCH_DIR / f"quality_{safe_name(model)}.jsonl"
            with qpath.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  → {qpath.relative_to(REPO_ROOT)}", flush=True)
            quality_summary[model] = analyze_quality_file(qpath)
            qs = quality_summary[model]
            print(
                f"  filters: docs kept {qs['doc_keep_rate']*100:.1f}%  "
                f"sents kept {qs['sent_keep_rate']*100:.1f}%  "
                f"({qs['n_sents']} sents from {qs['n_docs']} docs)",
                flush=True,
            )

        if not args.skip_throughput:
            mr: dict[int, dict] = {}
            for c in args.concurrencies:
                n_req = max(args.throughput_n, c * 2)
                print(f"  throughput c={c}, n={n_req} …", flush=True)
                r = throughput_bench(model, c, n_req, args.num_predict)
                mr[c] = r
                print(
                    f"    {r['tokens']} tok / {r['wall_seconds']:.1f}s = "
                    f"{r['tok_per_s']:.1f} tok/s "
                    f"({r['errors']} errors)",
                    flush=True,
                )
            throughput[model] = {str(k): v for k, v in mr.items()}

    if not args.skip_throughput:
        tpath = BENCH_DIR / "throughput.json"
        tpath.write_text(json.dumps(throughput, indent=2))
        print(f"\n→ {tpath.relative_to(REPO_ROOT)}", flush=True)

    if not args.skip_quality:
        spath = BENCH_DIR / "quality_summary.json"
        spath.write_text(json.dumps(quality_summary, indent=2))
        print(f"→ {spath.relative_to(REPO_ROOT)}", flush=True)

    # Final projection table.
    if not args.skip_throughput:
        print("\nProjections (best concurrency per model):")
        print(f"  {'model':<28} {'best tok/s':>10} {'→100M':>10} {'→1B':>10}")
        for model, by_c in throughput.items():
            best = max(by_c.values(), key=lambda r: r["tok_per_s"])
            ts = best["tok_per_s"]
            if ts <= 0:
                continue
            sec_100m = 1e8 / ts
            sec_1b = 1e9 / ts
            print(
                f"  {model:<28} {ts:>10.1f} "
                f"{sec_100m/86400:>8.1f}d "
                f"{sec_1b/86400:>8.1f}d"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
