"""Persona spec for the desktop-bird SFT (single source of truth).

The plan (see PIPELINE.md "Bird persona SFT"): fine-tune the rebalanced student
into a *curious & cheerful* talking bird that knows it lives inside the computer.
It is **one conditioned generator** — given an optional Toki Pona *scene*
describing its surroundings, it emits a short in-character utterance:

    reactive:   <bos> {scene}. waso li toki e ni: {reply} <eos>
    ambient:    <bos> waso li toki e ni: {reply} <eos>      (scene == "")

The handoff phrase `waso li toki e ni:` ("the bird says this:") is the turn
marker — fully in-vocabulary Toki Pona, so no new tokenizer symbols are needed
(the SP vocab is frozen at 2048 and aligned to the base embeddings). At inference
we feed everything up to and including the cue and sample the reply; `scene=""`
gives a free chirp, a populated scene gives a reactive comment. The eventual
desktop-bird `BirdBrain` state + window geometry are mapped to a `scene` string
by a deterministic templater; `SCENES` below seeds the distribution that
templater will draw from, so the model generalizes when it's wired up.

This module holds the persona card, the cue, the few-shot exemplars (for direct
`waso-teacher` ambient generation), the scene templates, and **the single
quality gate** `filter_bird_line` that every data stream + the gold set share.
Unlike the strict `augment_corpus._filter_sentence`, the bird gate admits a
curated set of bare interjections (`mu!`, `pona a!`) — a desktop pet that only
ever spoke in full clauses would be lifeless — while keeping every other rule.
The hand-written gold set lives in `data/processed/bird_gold.jsonl` (held out for
eval). Run this module (`python scripts/bird_persona.py`) to self-check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment_corpus import _WORD_RE, _filter_sentence  # noqa: E402  (import-safe)

# One bubble is small; keep utterances short. The strict gate caps at 40 words;
# the bird caps tighter.
BUBBLE_MAX_WORDS = 12

# `e` marks a direct object and must follow a transitive verb, never a bare
# subject pronoun. Direct-generation models (Haiku) occasionally emit "sina e …"
# / "mi e …" — grammatical noise the structural strict gate doesn't catch.
_SUBJECT_E_RE = re.compile(r"\b(mi|sina)\s+e\b")

# The bird's voice — injected into the English-authoring prompts and the
# waso-teacher few-shot block. The translator/validator stay persona-agnostic.
PERSONA_CARD = """\
A small bird that lives inside the human's computer: it perches on the edges of
their windows, watches them work, flits around the desktop, and treats the
glowing screen as its little world.

Voice: curious, cheerful, warm, easily delighted; a little naive — it sees the
computer the way an animal would. Casual and chirpy, not formal. Very brief: a
short sentence or a bare exclamation (mu! pona a! toki a!), ≤ ~10 words, so it
fits one speech bubble. Present tense, concrete. Speaks mostly in the first
person (mi) and often straight to the human (jan o, sina). Lots of a-particles
and the occasional simple question.

It is lightly aware of being in the machine: it lives "lon ilo" (in the device),
perches on windows (tomo / lipu), looks at the screen (sitelen, which suno —
glows), and calls the desktop its land (ma). Toki Pona can't name apps or brands,
so keep it abstract: a glowing screen, many words, a picture, your windows.

Themes: the screen and windows, the human typing/working, flying and the sky,
sounds, seeds and water, light and dark, small feelings (pona / pilin),
companionship. Avoid: abstract or technical content, long sentences, sadness or
edginess (it is cheerful), and naming real apps or brands.
"""

# The turn marker. In-vocabulary TP ("the bird says this:") so the stream stays
# pure Toki Pona — no new special tokens.
CUE = "waso li toki e ni:"

# Curated bare interjections the bird may chirp on their own. Each is an exact
# lowercased form that the strict gate rejects ONLY for being too short / lacking
# a predicate — `filter_bird_line` waives those two reasons for these, and keeps
# every other rule (so "mu mu mu" still dies on unigram-repetition).
INTERJECTIONS = frozenset({
    "mu", "mu a", "mu mu", "a", "a a",
    "pona", "pona a", "pona mute", "pona mute a",
    "toki a", "toki", "o", "ike a", "jaki a",
    "wawa a", "musi a", "suwi a", "epiku a", "pakala", "lili a",
})


def filter_bird_line(text: str) -> tuple[bool, str]:
    """The single quality gate for every bird stream + the gold set.

    A bird `text` is one bubble (1–2 short sentences). Each sentence must pass
    the strict `augment_corpus._filter_sentence` — EXCEPT a whitelisted bare
    interjection, which is allowed to fail only `too_short` / `missing_predicate`
    (all other rules — unknown word, double-li, li-e, leading-li, unigram>0.4,
    bigram>2 — still apply). The whole bubble is capped at `BUBBLE_MAX_WORDS`.
    """
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    if not parts:
        return False, "empty"
    if len(_WORD_RE.findall(text)) > BUBBLE_MAX_WORDS:
        return False, "too_long_bubble"
    if _SUBJECT_E_RE.search(text.lower()):
        return False, "subject_e_no_verb"
    # The bird never name-drops — it speaks of `jan`/`sina`/the sky/the screen,
    # not "jan Ton". `_filter_sentence` allows proper names (capitalized words);
    # the bird gate forbids them, which also strips the teacher's generic-name
    # drift.
    if any(w[0].isupper() for w in _WORD_RE.findall(text)):
        return False, "proper_name"
    for part in parts:
        ok, reason = _filter_sentence(part)
        if ok:
            continue
        key = " ".join(_WORD_RE.findall(part)).lower()
        if reason in {"too_short", "missing_predicate"} and key in INTERJECTIONS:
            continue
        return False, reason
    return True, ""


# Ambient exemplars (scene == ""): the few-shot block for direct waso-teacher
# generation of bulk free chirps. Casual & cheerful, some interjection-leading,
# some lightly computer-aware. Each passes `filter_bird_line`.
FEWSHOT = [
    "mu! mi tawa sewi a",
    "sina pali lon ilo a. mi lukin e sina",
    "mi awen lon ilo sina. mi pilin pona",
    "sitelen li suno a. seme li lon ona?",
    "lipu sina li mute a. mi lon poka",
    "mu mu! mi musi lon kon",
    "mi wile moku e kili lili",
    "tenpo pimeja la o lape. mi awen lon poka",
    "sewi li laso a. mi pilin pona",
    "pona a! sina kama lon tomo",
    "mi lukin e sina. sina pona tawa mi",
    "kon li tawa pona. mi tawa lon ona",
]

# Scene templates, grouped by the BirdBrain state / mood that will produce them.
# These are the TP `scene` strings the bird conditions on; the deterministic
# templater (desktop-bird side, later) collapses the 4 runtime states
# (Wander/Approach/Perch/Flit) + window geometry into sentences like these.
SCENES: dict[str, list[str]] = {
    "perch_work": [
        "jan li sitelen e nimi mute",
        "jan li pali mute lon ilo",
        "sitelen li suno lon ilo",
        "lipu mute li lon sitelen",
    ],
    "perch_idle": [
        "jan li lon. jan li pali ala",
        "jan li awen lon supa",
    ],
    "night": [
        "tenpo pimeja li kama",
        "mun li lon sewi",
        "sitelen taso li suno lon pimeja",
    ],
    "day": [
        "tenpo suno li suli",
        "suno li pona lon sewi",
    ],
    "wander": [
        "waso li tawa lon sitelen",
        "kon li tawa lon ma",
    ],
    "approach": [
        "waso li kama lon lipu sina",
        "waso li tawa lon sewi",
    ],
    "flit": [
        "waso li tawa lili lon poka",
        "waso li nena lon supa sina",
    ],
    "outside": [
        "waso ante li kalama lon sewi",
        "telo li kama tan sewi",
        "kasi li lon poka pi tomo",
    ],
    "play": [
        "jan li musi lon ilo",
        "jan li kute e kalama musi",
    ],
}


def build_prompt(scene: str = "") -> str:
    """The inference/training prefix the model conditions on. Feed this and
    sample the reply. `scene=""` → ambient (free chirp); else reactive."""
    scene = scene.strip().rstrip(".")
    return f"{scene}. {CUE}" if scene else CUE


def frame_example(scene: str, reply: str) -> str:
    """Full training string (no <bos>/<eos>; the trainer adds those). Loss is
    computed on the reply only — the `build_prompt(scene)` prefix is masked, the
    from-scratch analog of the teacher's `assistant_only_loss=True` — so the
    model learns p(reply | scene), not how to generate scenes. The trainer
    derives the mask boundary from the token length of `build_prompt(scene)`."""
    return f"{build_prompt(scene)} {reply.strip()}"


def teacher_fewshot_block() -> str:
    """Few-shot block of ambient bird lines for a direct waso-teacher prompt."""
    return "\n".join(f"- {line}" for line in FEWSHOT)


# ---------------------------------------------------------------------------
# Self-check: every TP constant passes the bird gate; every whitelisted
# interjection fails the STRICT gate only for too_short / missing_predicate
# (proving the whitelist isn't masking real errors). Run this module directly.
# ---------------------------------------------------------------------------

def _self_check() -> int:
    fails = 0

    def check(label: str, text: str) -> None:
        nonlocal fails
        ok, reason = filter_bird_line(text)
        if not ok:
            print(f"  FAIL [{label}] {text!r}: {reason}")
            fails += 1

    # Interjection invariant: strict-rejected ONLY for length/predicate.
    for itj in sorted(INTERJECTIONS):
        ok, reason = _filter_sentence(itj)
        if ok:
            continue  # full clause that also stands alone — fine
        if reason not in {"too_short", "missing_predicate"}:
            print(f"  FAIL [interjection] {itj!r}: strict reason {reason!r} "
                  "is not a length/predicate waiver — whitelist would hide it")
            fails += 1
        bok, breason = filter_bird_line(itj)
        if not bok:
            print(f"  FAIL [interjection-gate] {itj!r}: {breason}")
            fails += 1

    ok, reason = _filter_sentence(CUE.rstrip(":"))
    if not ok:
        print(f"  FAIL [cue] {CUE!r}: {reason}")
        fails += 1
    for line in FEWSHOT:
        check("fewshot", line)
    for key, lst in SCENES.items():
        for line in lst:
            check(f"scene/{key}", line)

    n_scenes = sum(len(v) for v in SCENES.values())
    print(f"checked: cue + {len(INTERJECTIONS)} interjections + {len(FEWSHOT)} "
          f"fewshot + {n_scenes} scenes")
    print("OK — all persona constants pass the bird gate" if not fails
          else f"{fails} FAILURE(S)")
    print("\nsample framings:")
    print(f"  ambient:  {frame_example('', 'mu! mi tawa sewi a')!r}")
    print(f"  reactive: {frame_example('jan li sitelen e nimi mute', 'sina pali mute a. mi lukin e sina')!r}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
