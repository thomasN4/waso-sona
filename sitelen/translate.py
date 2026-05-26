"""Bidirectional translator between Latin-script Toki Pona and sitelen pona UCSUR.

This module is an application-layer concern: the language model trains and runs
on Latin-script Toki Pona, and this translator deterministically converts
UCSUR user input to Latin for the model and Latin model output back to UCSUR
for display.
"""
from __future__ import annotations

import re

from .glyphs import (
    ASCII_PUNCT_MAP,
    CARTOUCHE_END, CARTOUCHE_EXT_END, CARTOUCHE_EXT_START, CARTOUCHE_START,
    CODEPOINT_TO_WORD,
    END_REVERSE_LONG_GLYPH, LONG_GLYPH_EXTENSION,
    KU_SULI_WORDS, MIDDLE_COLON, MIDDLE_DOT,
    PU_WORDS, SCALING_JOINER, START_REVERSE_LONG_GLYPH, STACKING_JOINER,
    WORD_TO_CODEPOINT,
)
from .syllabify import syllabify

_CARTOUCHE_OPENERS = {CARTOUCHE_START, CARTOUCHE_EXT_START}
_CARTOUCHE_CLOSERS = {CARTOUCHE_END, CARTOUCHE_EXT_END}
_JOINERS = {STACKING_JOINER, SCALING_JOINER}
_STRIPPED = {
    LONG_GLYPH_EXTENSION, START_REVERSE_LONG_GLYPH, END_REVERSE_LONG_GLYPH,
}


def _build_syllable_to_word() -> dict[str, str]:
    """Map each TP-phonotactic syllable to a representative TP word.

    Among words whose first syllable equals the target, prefer pu over ku
    (pu words dominate actual usage and produce more recognizable
    cartouches), then shorter words, then alphabetic order. The choice is
    deterministic so encode/decode round-trips cleanly: encoding emits
    glyph(W); decoding reads glyph(W) back as `syllabify(W)[0]`.
    """
    pu = set(PU_WORDS)
    by_syllable: dict[str, list[str]] = {}
    for word in PU_WORDS + KU_SULI_WORDS:
        sylls = syllabify(word)
        if sylls:
            by_syllable.setdefault(sylls[0], []).append(word)
    return {
        syl: sorted(words, key=lambda w: (0 if w in pu else 1, len(w), w))[0]
        for syl, words in by_syllable.items()
    }


_SYLLABLE_TO_WORD = _build_syllable_to_word()


def _glyphs_for_syllable(syl: str) -> list[str]:
    """Return the UCSUR glyph(s) that represent a TP syllable inside a cartouche.

    Most syllables map to a single glyph (the representative word's). CVN
    syllables that have no representative word (e.g. "son", "jon", "lan") fall
    back to CV + n: emit the CV-word glyph followed by the "n" glyph.
    """
    word = _SYLLABLE_TO_WORD.get(syl)
    if word is not None:
        return [chr(WORD_TO_CODEPOINT[word])]
    if len(syl) > 1 and syl.endswith("n"):
        cv = _SYLLABLE_TO_WORD.get(syl[:-1])
        n = _SYLLABLE_TO_WORD.get("n")
        if cv and n:
            return [chr(WORD_TO_CODEPOINT[cv]), chr(WORD_TO_CODEPOINT[n])]
    return []


# ---------------------------------------------------------------------------
# UCSUR -> Latin
# ---------------------------------------------------------------------------

def ucsur_to_latin(text: str) -> str:
    """Translate sitelen pona UCSUR text into Latin-script Toki Pona."""
    tokens: list[str] = []
    cartouche: list[str] | None = None

    def flush() -> None:
        nonlocal cartouche
        if cartouche is None:
            return
        name = "".join(cartouche)
        if name:
            tokens.append(name[0].upper() + name[1:])
        cartouche = None

    for ch in text:
        cp = ord(ch)
        if cp in _CARTOUCHE_OPENERS:
            flush()
            cartouche = []
        elif cp in _CARTOUCHE_CLOSERS:
            flush()
        elif cp in _STRIPPED or cp in _JOINERS:
            continue
        elif cp == MIDDLE_DOT:
            if cartouche is None:
                tokens.append(".")
        elif cp == MIDDLE_COLON:
            if cartouche is None:
                tokens.append(":")
        elif cp in CODEPOINT_TO_WORD:
            word = CODEPOINT_TO_WORD[cp]
            if cartouche is not None:
                sylls = syllabify(word)
                if sylls:
                    cartouche.append(sylls[0])
            else:
                tokens.append(word)
        # Unrecognized chars are dropped; UCSUR input is assumed clean.
    flush()

    out: list[str] = []
    for tok in tokens:
        if tok in {".", ":"} and out:
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Latin -> UCSUR
# ---------------------------------------------------------------------------

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+|[.!?:,]")


def latin_to_ucsur(text: str) -> str:
    """Translate Latin-script Toki Pona into sitelen pona UCSUR text."""
    pieces: list[str] = []
    for m in _LATIN_TOKEN_RE.finditer(text):
        piece = m.group(0)
        if piece == ",":
            continue
        if piece in ASCII_PUNCT_MAP:
            pieces.append(chr(ASCII_PUNCT_MAP[piece]))
            continue
        pieces.extend(_translate_word(piece))
    return _join_ucsur(pieces)


def _translate_word(word: str) -> list[str]:
    lower = word.lower()
    is_name = word[0].isupper() and lower not in WORD_TO_CODEPOINT
    if not is_name:
        cp = WORD_TO_CODEPOINT.get(lower)
        return [chr(cp)] if cp is not None else []
    out = [chr(CARTOUCHE_START)]
    for syl in syllabify(lower):
        out.extend(_glyphs_for_syllable(syl))
    out.append(chr(CARTOUCHE_END))
    return out


def _join_ucsur(pieces: list[str]) -> str:
    """Render glyph pieces with spaces between words; none inside cartouches."""
    out: list[str] = []
    in_cartouche = False
    for g in pieces:
        cp = ord(g) if len(g) == 1 else -1
        is_punct = cp in (MIDDLE_DOT, MIDDLE_COLON)
        if cp == CARTOUCHE_START or cp == CARTOUCHE_EXT_START:
            if out:
                out.append(" ")
            out.append(g)
            in_cartouche = True
        elif cp == CARTOUCHE_END or cp == CARTOUCHE_EXT_END:
            out.append(g)
            in_cartouche = False
        elif in_cartouche:
            out.append(g)
        elif is_punct:
            out.append(g)
        else:
            if out:
                out.append(" ")
            out.append(g)
    return "".join(out)
