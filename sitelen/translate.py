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
    MIDDLE_COLON, MIDDLE_DOT,
    SCALING_JOINER, START_REVERSE_LONG_GLYPH, STACKING_JOINER,
    WORD_TO_CODEPOINT,
)

_CARTOUCHE_OPENERS = {CARTOUCHE_START, CARTOUCHE_EXT_START}
_CARTOUCHE_CLOSERS = {CARTOUCHE_END, CARTOUCHE_EXT_END}
_JOINERS = {STACKING_JOINER, SCALING_JOINER}
_STRIPPED = {
    LONG_GLYPH_EXTENSION, START_REVERSE_LONG_GLYPH, END_REVERSE_LONG_GLYPH,
}

# Inside a cartouche, sitelen pona spells a name acrostically: each glyph stands
# for the FIRST LETTER of the Toki Pona word it normally represents. This is the
# convention used in pu (one glyph per letter), not a syllabic one. _LETTER_TO_WORD
# picks one representative word per Toki Pona letter for *encoding*; *decoding*
# reads any glyph back as its word's first letter, so the round-trip holds for
# every letter no matter which representatives are chosen here. The set below
# deliberately avoids grammatical particles (a, e, o, pi, li, la) as glyphs.
_LETTER_TO_WORD = {
    "a": "ala", "e": "esun", "i": "ilo", "o": "open", "u": "uta",
    "p": "pona", "t": "toki", "k": "kala", "s": "suno", "m": "mama",
    "n": "nasin", "l": "luka", "j": "jan", "w": "waso",
}

_LETTER_TO_GLYPH = {
    letter: chr(WORD_TO_CODEPOINT[word])
    for letter, word in _LETTER_TO_WORD.items()
}


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
                cartouche.append(word[0])  # acrostic: the glyph's first letter
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

# Punctuation policy, following common sitelen pona usage (sona pona):
# punctuation mainly marks sentence boundaries, and most writers mark one with
# a plain space (or line break), not a glyph. So ./!/? become a single space;
# the period/seme/exclamation glyphs are avoided. Commas are not written at
# all. The colon is the one mark that keeps a dedicated glyph (middle colon),
# which sitelen pona does use before quotes and lists.
_SENTENCE_ENDERS = {".", "!", "?"}
_SENTENCE_BOUNDARY = " "


def latin_to_ucsur(text: str) -> str:
    """Translate Latin-script Toki Pona into sitelen pona UCSUR text.

    Word-glyphs are written with no separating spaces; the only space is a
    sentence boundary. Because that boundary is a bare space and the decoder
    treats spaces as non-significant (glyphs are self-delimiting), the
    Latin -> UCSUR -> Latin round trip drops sentence-final punctuation by
    design -- which is how sitelen pona itself behaves.
    """
    pieces: list[str] = []
    for m in _LATIN_TOKEN_RE.finditer(text):
        piece = m.group(0)
        if piece in _SENTENCE_ENDERS:
            pieces.append(_SENTENCE_BOUNDARY)
        elif piece == ",":
            continue  # commas are not written in sitelen pona
        elif piece in ASCII_PUNCT_MAP:
            # Only ":" reaches here; ./!/? are intercepted as boundaries above.
            pieces.append(chr(ASCII_PUNCT_MAP[piece]))
        else:
            pieces.extend(_translate_word(piece))
    return _join_ucsur(pieces)


def _translate_word(word: str) -> list[str]:
    lower = word.lower()
    is_name = word[0].isupper() and lower not in WORD_TO_CODEPOINT
    if not is_name:
        cp = WORD_TO_CODEPOINT.get(lower)
        return [chr(cp)] if cp is not None else []
    out = [chr(CARTOUCHE_START)]
    for letter in lower:
        glyph = _LETTER_TO_GLYPH.get(letter)
        if glyph is not None:
            out.append(glyph)
    out.append(chr(CARTOUCHE_END))
    return out


def _join_ucsur(pieces: list[str]) -> str:
    """Concatenate glyph pieces into a sitelen pona string.

    Word-glyphs and cartouches run together with no separators -- UCSUR glyphs
    are self-delimiting, so the spacing the Latin script needs is unnecessary.
    The only space is a sentence boundary (the _SENTENCE_BOUNDARY sentinel).
    Runs of boundaries, and any leading or trailing boundary, are collapsed so
    the result holds at most a single space between sentences and none at the
    edges.
    """
    out: list[str] = []
    at_boundary = True  # treat start-of-text as a boundary to swallow leading spaces
    for g in pieces:
        if g == _SENTENCE_BOUNDARY:
            if not at_boundary:
                out.append(_SENTENCE_BOUNDARY)
                at_boundary = True
        else:
            out.append(g)
            at_boundary = False
    if out and out[-1] == _SENTENCE_BOUNDARY:
        out.pop()  # trailing boundary
    return "".join(out)
