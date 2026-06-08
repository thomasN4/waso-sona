"""Translate the authored English bird pairs into Toki Pona (Phase A4).

Each `bird_english.jsonl` record `{scene_en, reply_en, mood, source}` is turned
into a TP `{ctx, text, mood, source:"translate"}` by translating BOTH sides
through the Ollama-served `waso-translator` (the proven Stage-2-v2 lever): the
reply becomes the bird's utterance, the scene becomes the conditioning context.
Ambient pairs (`scene_en == ""`) translate only the reply and keep `ctx == ""`.

Reuses `translate_corpus`/`augment_corpus` wholesale — `_ollama_generate` (chat
endpoint, so the translator's chat template wraps the prompt), `_translate_prompt`,
`_split_into_sentences` — plus the resumable `.done` sidecar + global cross-run
dedup + ThreadPool pattern. The single quality gate is `bird_persona.filter_bird_line`
(so bird interjections survive and the bubble cap applies), and translated scenes
must stay clauses (contain `li` or a `mi/sina/o` subject) so the conditioning
distribution matches `SCENES` rather than collapsing to fragments.

Two Ollama calls per reactive pair → set `OLLAMA_NUM_PARALLEL`
(scripts/enable_ollama_parallel.sh) and a matching `--concurrency`, or the server
serializes and concurrency does nothing (see translate_corpus).

Output: data/processed/bird_translate.jsonl (appended) — {ctx, text, mood, source}.

Usage::

    python scripts/translate_bird.py --max-pairs 20 --concurrency 4   # smoke
    python scripts/translate_bird.py --concurrency 16                 # bulk
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

import augment_corpus as ac  # noqa: E402
import bird_persona as bp  # noqa: E402

DEFAULT_MODEL = "waso-translator"
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "bird_english.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "bird_translate.jsonl"

_SUBJECTS = {"mi", "sina", "o"}


def _translate_en(eng: str, model: str, max_tokens: int, repeat_penalty: float,
                  repeat_last_n: int, fewshot: bool) -> tuple[str | None, str, int]:
    """Translate one English sentence → first TP sentence. (hyp|None, reason, toks)."""
    text, toks = ac._ollama_generate(
        ac._translate_prompt(eng, fewshot=fewshot), model, max_tokens,
        repeat_penalty, repeat_last_n)
    sents = ac._split_into_sentences(text)
    if not sents:
        return None, "empty", toks
    hyp = sents[0]
    if hyp.lower().strip(".!? ") == eng.lower().strip(".!? "):
        return None, "copied_source", toks
    return hyp, "", toks


def _is_clause(tp: str) -> bool:
    words = [w.lower() for w in ac._WORD_RE.findall(tp)]
    return "li" in words or any(w in _SUBJECTS for w in words)


def _translate_pair(pair: dict, model: str, max_tokens: int,
                    repeat_penalty: float, repeat_last_n: int, fewshot: bool
                    ) -> tuple[dict | None, str, int]:
    """Translate a {scene_en, reply_en} pair. (record|None, reason, total_toks)."""
    toks = 0
    # reply (always)
    reply_tp, reason, t = _translate_en(pair["reply_en"], model, max_tokens,
                                        repeat_penalty, repeat_last_n, fewshot)
    toks += t
    if reply_tp is None:
        return None, f"reply_{reason}", toks
    ok, why = bp.filter_bird_line(reply_tp)
    if not ok:
        return None, f"reply_{why}", toks
    # scene (only if reactive)
    scene_tp = ""
    if pair.get("scene_en"):
        scene_tp, reason, t = _translate_en(pair["scene_en"], model, max_tokens,
                                            repeat_penalty, repeat_last_n, fewshot)
        toks += t
        if scene_tp is None:
            return None, f"scene_{reason}", toks
        ok, why = bp.filter_bird_line(scene_tp)
        if not ok:
            return None, f"scene_{why}", toks
        if not _is_clause(scene_tp):
            return None, "scene_fragment", toks
        if reply_tp.lower().strip() == scene_tp.lower().strip():
            return None, "reply_copies_scene", toks
    return ({"ctx": scene_tp, "text": reply_tp, "mood": pair.get("mood", ""),
             "source": f"{model}/translate"}, "", toks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit-output-tokens", type=int, default=32)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--repeat-last-n", type=int, default=64)
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input} — run build_bird_english.py "
              "first", file=sys.stderr)
        return 1

    pairs = [json.loads(ln) for ln in
             args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Loaded {len(pairs):,} English pairs from {args.input.name}.")

    # Resumability: .done keys each pair by (scene_en \t reply_en).
    def pair_key(p: dict) -> str:
        return f"{p.get('scene_en','')}\t{p['reply_en']}"

    done_path = args.output.with_suffix(args.output.suffix + ".done")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if done_path.exists():
        with done_path.open(encoding="utf-8") as fh:
            done = {ln.rstrip("\n") for ln in fh if ln.strip()}
    remaining = [p for p in pairs if pair_key(p) not in done]
    print(f"Already done: {len(done):,}.  To process: {len(remaining):,}.")
    if not remaining:
        print("Nothing to do — re-running is a no-op.")
        return 0

    # Global cross-run dedup on (ctx, text).
    global_seen: set[tuple[str, str]] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    r = json.loads(raw)
                    global_seen.add((r["ctx"].lower().strip(),
                                     r["text"].lower().strip()))
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"Preloaded {len(global_seen):,} existing TP pairs for global dedup.")

    rejects: collections.Counter = collections.Counter()
    accepted = global_dups = total_tokens = 0
    start = time.monotonic()
    out_fh = args.output.open("a", encoding="utf-8")
    done_fh = done_path.open("a", encoding="utf-8")

    def _work(p: dict):
        try:
            return _translate_pair(p, args.model, args.limit_output_tokens,
                                   args.repeat_penalty, args.repeat_last_n,
                                   args.fewshot)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return None, f"api_error:{type(exc).__name__}", 0

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        fut_to_pair = {pool.submit(_work, p): p for p in remaining}
        for fut in concurrent.futures.as_completed(fut_to_pair):
            p = fut_to_pair[fut]
            n_done += 1
            record, reason, toks = fut.result()
            total_tokens += toks
            if record is not None:
                key = (record["ctx"].lower().strip(), record["text"].lower().strip())
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
            done_fh.write(pair_key(p) + "\n")
            done_fh.flush()
            if n_done % 25 == 0 or n_done == len(remaining):
                elapsed = time.monotonic() - start
                rate = total_tokens / elapsed if elapsed > 0 else 0.0
                print(f"[{n_done}/{len(remaining)}] accepted={accepted} "
                      f"dups={global_dups} tok/s={rate:.0f}", flush=True)

    out_fh.close()
    done_fh.close()

    n = len(remaining)
    print("\n=== Bird translation summary ===")
    print(f"  Processed:   {n:,}")
    print(f"  Accepted:    {accepted:,} ({100*accepted/max(n,1):.1f}%)")
    print(f"  Global dups: {global_dups:,}")
    print("  Reject reasons:")
    for reason, c in rejects.most_common():
        print(f"    {reason:24s} {c:,}")
    print(f"  → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
