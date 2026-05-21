"""Tests for SitelenTokenizer."""
import tempfile

import pytest

from tokenizer import SitelenTokenizer, WORD_TO_CODEPOINT, syllabify
from tokenizer.glyphs import (
    CARTOUCHE_END, CARTOUCHE_START, MIDDLE_COLON, MIDDLE_DOT,
)


@pytest.fixture
def tok() -> SitelenTokenizer:
    return SitelenTokenizer()


# ---- vocabulary ----

def test_vocab_size_matches_layout(tok: SitelenTokenizer) -> None:
    # 4 specials + 120 pu + 17 ku suli + 11 structural punctuation
    assert tok.vocab_size == 4 + 120 + 17 + 11


def test_every_word_round_trips_through_vocab(tok: SitelenTokenizer) -> None:
    for word, cp in WORD_TO_CODEPOINT.items():
        glyph = chr(cp)
        assert glyph in tok.token_to_id
        assert tok.id_to_token[tok.token_to_id[glyph]] == glyph


# ---- syllabifier ----

@pytest.mark.parametrize("word,expected", [
    ("mali", ["ma", "li"]),
    ("toki", ["to", "ki"]),
    ("sonja", ["son", "ja"]),
    ("anpa", ["an", "pa"]),
    ("kijetesantakalu", ["ki", "je", "te", "san", "ta", "ka", "lu"]),
    ("a", ["a"]),
    ("nnn", ["n", "n", "n"]),  # degenerate: no vowel
])
def test_syllabify(word: str, expected: list[str]) -> None:
    assert syllabify(word) == expected


# ---- tokenize ----

def test_known_words_become_single_glyphs(tok: SitelenTokenizer) -> None:
    glyphs = tok.tokenize("mi olin e sina")
    assert glyphs == [
        chr(WORD_TO_CODEPOINT["mi"]),
        chr(WORD_TO_CODEPOINT["olin"]),
        chr(WORD_TO_CODEPOINT["e"]),
        chr(WORD_TO_CODEPOINT["sina"]),
    ]


def test_period_maps_to_middle_dot(tok: SitelenTokenizer) -> None:
    glyphs = tok.tokenize("mi pona.")
    assert glyphs[-1] == chr(MIDDLE_DOT)


def test_capitalized_name_becomes_cartouche(tok: SitelenTokenizer) -> None:
    glyphs = tok.tokenize("jan Mali li pona")
    # Expect: jan, [, ma-glyph, la-glyph, ], li, pona
    assert glyphs[0] == chr(WORD_TO_CODEPOINT["jan"])
    assert glyphs[1] == chr(CARTOUCHE_START)
    assert glyphs[2] == chr(WORD_TO_CODEPOINT["ma"])   # "m" -> ma
    assert glyphs[3] == chr(WORD_TO_CODEPOINT["la"])   # "l" -> la
    assert glyphs[4] == chr(CARTOUCHE_END)
    assert glyphs[5] == chr(WORD_TO_CODEPOINT["li"])
    assert glyphs[6] == chr(WORD_TO_CODEPOINT["pona"])


def test_unknown_word_is_syllabified(tok: SitelenTokenizer) -> None:
    # "sonja" is not a TP word -> cartouche of [s][j] glyphs
    glyphs = tok.tokenize("sonja")
    assert glyphs[0] == chr(CARTOUCHE_START)
    assert glyphs[1] == chr(WORD_TO_CODEPOINT["sama"])  # s -> sama
    assert glyphs[2] == chr(WORD_TO_CODEPOINT["jaki"])  # j -> jaki
    assert glyphs[-1] == chr(CARTOUCHE_END)


def test_comma_is_dropped(tok: SitelenTokenizer) -> None:
    assert "," not in "".join(tok.tokenize("mi, sina"))


def test_colon_maps_to_middle_colon(tok: SitelenTokenizer) -> None:
    glyphs = tok.tokenize("toki:")
    assert glyphs[-1] == chr(MIDDLE_COLON)


# ---- encode/decode ----

def test_encode_wraps_with_bos_eos(tok: SitelenTokenizer) -> None:
    ids = tok.encode("mi pona")
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id


def test_encode_without_special_tokens(tok: SitelenTokenizer) -> None:
    ids = tok.encode("mi pona", add_special_tokens=False)
    assert tok.bos_token_id not in ids
    assert tok.eos_token_id not in ids
    assert len(ids) == 2


def test_decode_produces_valid_ucsur(tok: SitelenTokenizer) -> None:
    text = "jan Mali li pona."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # Every char should be UCSUR (sitelen pona block) or space.
    for ch in decoded:
        if ch == " ":
            continue
        cp = ord(ch)
        assert 0xF1900 <= cp <= 0xF19FF, f"non-UCSUR char: {ch!r} (U+{cp:X})"


def test_encode_roundtrip_known_sentence(tok: SitelenTokenizer) -> None:
    ids = tok.encode("mi olin e sina", add_special_tokens=False)
    decoded = tok.decode(ids)
    assert decoded == " ".join([
        chr(WORD_TO_CODEPOINT["mi"]),
        chr(WORD_TO_CODEPOINT["olin"]),
        chr(WORD_TO_CODEPOINT["e"]),
        chr(WORD_TO_CODEPOINT["sina"]),
    ])


# ---- persistence ----

def test_save_and_load_roundtrip(tok: SitelenTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        loaded = SitelenTokenizer.from_pretrained(tmp)
        assert loaded.get_vocab() == tok.get_vocab()
        assert loaded.encode("mi pona") == tok.encode("mi pona")
