"""Generate bird-persona Toki Pona directly with Claude Haiku 4.5 (Phase A5, v2).

The probe (2026-06-08) showed Haiku writes valid in-persona TP at ~82% gate
pass-rate with near-zero word invention — far better persona fidelity than the
`waso-teacher` (generic-TP) stream and with no English→translate round-trip. So
Haiku is the primary bulk generator, in two modes:

- **ambient** (`ctx == ""`): free chirps, themed batches for variety.
- **reactive** (`ctx == <scene>`): for each `bird_persona.SCENES` entry, replies
  reacting to that exact scene — anchoring the conditioning distribution to what
  the desktop-bird templater will emit.

Every candidate passes `bird_persona.filter_bird_line` (incl. the tightened
`subject_e_no_verb` rule); a global cross-run dedup keeps the corpus distinct.
Reruns extend the output. Needs `ANTHROPIC_API_KEY` in the environment.

Output: data/processed/bird_haiku.jsonl (appended) — {ctx, text, mood, source}.

Usage::

    python scripts/gen_bird_haiku.py --ambient-batches 2 --reactive-reps 1   # smoke
    python scripts/gen_bird_haiku.py --ambient-batches 30 --reactive-reps 4  # bulk
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import anthropic  # noqa: E402
import bird_persona as bp  # noqa: E402
from sitelen.glyphs import WORD_TO_CODEPOINT  # noqa: E402

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "bird_haiku.jsonl"

_VOCAB = " ".join(sorted(WORD_TO_CODEPOINT))

SYSTEM = f"""You write single short lines of Toki Pona for a desktop-pet bird: a small, \
curious, cheerful bird that lives inside the user's computer, perches on their windows, \
watches them work, and loves the sky. Casual and chirpy, never formal.

Hard rules (lines that break them are discarded, so follow them exactly):
1. Use ONLY these {len(WORD_TO_CODEPOINT)} words, all lowercase, nothing else — no names, \
no invented words:
{_VOCAB}
2. Each line is one short utterance, 3 to 10 words. Present tense, concrete, cheerful.
3. Grammar: `subject li predicate`; `e` marks a direct object and MUST come right after a \
verb (never after a bare `mi`/`sina`); `la` after a context phrase; `pi` regroups \
modifiers. After the subjects `mi` / `sina` / `o` you DROP `li`.
4. Never write `li li`, never `li e`, never `mi e`/`sina e`, never start a line with `li`.
5. Every line needs a predicate: it contains `li`, OR it contains `mi`, `sina`, or `o`.

Computer-aware imagery is good but stay abstract (Toki Pona can't name apps): the screen \
is `sitelen` (it `suno` — glows), windows are `lipu`/`tomo`, the device is `ilo`, the \
desktop is `ma`. Still a bird: sky `sewi`, wind `kon`, seeds `kili`, sounds `kalama`.

Example lines:
{chr(10).join(bp.FEWSHOT)}

Output ONLY Toki Pona lines, one per line, no numbering, no English, no commentary."""

# Per-batch theme hints diversify the ambient stream beyond temperature alone.
_THEMES = [
    "the glowing screen, windows, and the words the human types",
    "the sky, flying, wind, and other birds",
    "talking straight to the human (greetings, affection, little questions)",
    "seeds, water, food, and small happy feelings",
    "day and night, light and dark, resting",
    "tiny exclamations and chirps (a, mu a, pona a) attached to a short thought",
]


def _parse_lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("-*0123456789.) ").strip()
        if ln:
            out.append(ln)
    return out


def _ambient_task(client, model: str, temperature: float, theme: str) -> list[dict]:
    msg = client.messages.create(
        model=model, max_tokens=1024, temperature=temperature, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Write 20 new bird lines, focused on: {theme}."}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    recs = []
    for ln in _parse_lines(text):
        ok, _ = bp.filter_bird_line(ln)
        if ok:
            recs.append({"ctx": "", "text": ln, "mood": "ambient",
                         "source": f"{model}/ambient"})
    return recs


def _reactive_task(client, model: str, temperature: float, scene: str, mood: str,
                   k: int) -> list[dict]:
    msg = client.messages.create(
        model=model, max_tokens=512, temperature=temperature, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f'The scene around the bird right now is: "{scene}". Write {k} '
                   "short bird replies reacting to it, one per line, only Toki Pona."}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    recs = []
    for ln in _parse_lines(text):
        ok, _ = bp.filter_bird_line(ln)
        if ok and ln.lower().strip() != scene.lower().strip():
            recs.append({"ctx": scene, "text": ln, "mood": mood,
                         "source": f"{model}/reactive"})
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--ambient-batches", type=int, default=24,
                    help="ambient calls (each asks for ~20 lines)")
    ap.add_argument("--reactive-reps", type=int, default=4,
                    help="replies requested per scene in SCENES")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)

    client = anthropic.Anthropic()

    # Build the task list: ambient batches (rotating themes) + one per scene.
    tasks = []
    for i in range(args.ambient_batches):
        tasks.append(("ambient", _THEMES[i % len(_THEMES)], None))
    for mood, scenes in bp.SCENES.items():
        for scene in scenes:
            tasks.append(("reactive", scene, mood))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    global_seen: set[tuple[str, str]] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    r = json.loads(raw)
                    global_seen.add((r.get("ctx", "").lower().strip(),
                                     r["text"].lower().strip()))
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"Preloaded {len(global_seen):,} existing lines for global dedup. "
          f"{len(tasks)} tasks.", flush=True)

    def _run(task):
        kind, a, b = task
        try:
            if kind == "ambient":
                return _ambient_task(client, args.model, args.temperature, a)
            return _reactive_task(client, args.model, args.temperature, a, b,
                                  args.reactive_reps)
        except anthropic.APIError as exc:
            print(f"  ! api error: {exc}", file=sys.stderr)
            return []

    accepted = dups = 0
    by_source: Counter = Counter()
    out_fh = args.output.open("a", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for n, recs in enumerate(pool.map(_run, tasks), 1):
            for r in recs:
                key = (r["ctx"].lower().strip(), r["text"].lower().strip())
                if key in global_seen:
                    dups += 1
                    continue
                global_seen.add(key)
                out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                accepted += 1
                by_source[r["source"].split("/")[-1]] += 1
            if n % 10 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] accepted={accepted} dups={dups}",
                      flush=True)
    out_fh.close()

    print("\n=== Haiku bird summary ===")
    print(f"  Accepted: {accepted:,}  (ambient {by_source['ambient']} / "
          f"reactive {by_source['reactive']})")
    print(f"  Dups skipped: {dups:,}")
    print(f"  → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
