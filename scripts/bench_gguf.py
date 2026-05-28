"""Compare Ollama-served GGUF teacher candidates on Toki Pona generation.

Generates ~`--target-tokens` per model from continuation prompts via Ollama
`/api/chat` (which applies each model's baked-in chat template), then scores
with the same validity + distinct-n metrics as `bench_adapters`. Used to pick
the Stage-2 teacher among the merged GGUF models.

The Ollama server must be running. Raw generations are saved per model.

Usage::

    python scripts/bench_gguf.py
    python scripts/bench_gguf.py --models waso-baseline waso-parity --target-tokens 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402
from bench_adapters import SEEDS, score_texts  # noqa: E402

OLLAMA_CHAT = "http://localhost:11434/api/chat"
BENCH_DIR = REPO_ROOT / "data" / "bench"
DEFAULT_MODELS = ["waso-baseline", "waso-parity", "waso-bumped"]


def chat(model: str, prompt: str, num_predict: int, temperature: float,
         seed: int) -> tuple[str, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": 0.95,
            "seed": seed,
        },
    }
    req = urllib.request.Request(
        OLLAMA_CHAT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d.get("message", {}).get("content", "").strip(), d.get("eval_count", 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--target-tokens", type=int, default=10000)
    ap.add_argument("--num-predict", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args(argv)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        texts: list[str] = []
        records: list[dict] = []
        tokens = 0
        i = 0
        while tokens < args.target_tokens:
            seed = SEEDS[i % len(SEEDS)]
            prompt = augment_corpus._continuation_prompt(seed)
            try:
                out, n = chat(model, prompt, args.num_predict, args.temperature, seed=i)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  ! error: {e}", file=sys.stderr)
                break
            texts.append(out)
            records.append({"i": i, "seed": seed, "tokens": n, "text": out})
            tokens += n if n else len(out.split())
            i += 1
            if i % 10 == 0:
                print(f"  {i} gens, ~{tokens} tokens", flush=True)
        outpath = BENCH_DIR / f"gguf_{model.replace(':', '_')}.jsonl"
        with outpath.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        m = score_texts(texts)
        m["total_tokens"] = tokens
        m["n_gens"] = i
        summary[model] = m
        print(f"  done: {i} gens, ~{tokens} tokens → {outpath.relative_to(REPO_ROOT)}",
              flush=True)

    (BENCH_DIR / "gguf_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("distinct-n: higher = less repetition (1.0 = no repeats)")
    hdr = ("model", "tokens", "doc_keep", "sent_keep", "strict", "d-1", "d-2", "d-3", "avg_w")
    print("  " + " ".join(f"{h:>11}" for h in hdr))
    for model, m in summary.items():
        print("  " + " ".join(f"{v:>11}" for v in (
            f"{model:>11}",
            f"{m['total_tokens']:>11}",
            f"{m['doc_keep_rate']*100:.0f}%",
            f"{m['sent_keep_rate']*100:.0f}%",
            f"{m['strict_pass_rate']*100:.0f}%",
            f"{m['distinct_1']:.2f}",
            f"{m['distinct_2']:.2f}",
            f"{m['distinct_3']:.2f}",
            f"{m['avg_words']:.0f}",
        )))
    print(f"\n→ {(BENCH_DIR / 'gguf_summary.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
