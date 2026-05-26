"""Tests for the Latin <-> sitelen pona UCSUR translator."""
import pytest

from sitelen import latin_to_ucsur, syllabify, ucsur_to_latin
from sitelen.glyphs import (
    CARTOUCHE_END, CARTOUCHE_EXT_END, CARTOUCHE_EXT_START, CARTOUCHE_START,
    END_REVERSE_LONG_GLYPH, LONG_GLYPH_EXTENSION, MIDDLE_COLON, MIDDLE_DOT,
    SCALING_JOINER, STACKING_JOINER, START_REVERSE_LONG_GLYPH,
    WORD_TO_CODEPOINT,
)


def _g(word: str) -> str:
    return chr(WORD_TO_CODEPOINT[word])


# ---- syllabifier ----

@pytest.mark.parametrize("word,expected", [
    ("mali", ["ma", "li"]),
    ("toki", ["to", "ki"]),
    ("sonja", ["son", "ja"]),
    ("anpa", ["an", "pa"]),
    ("kijetesantakalu", ["ki", "je", "te", "san", "ta", "ka", "lu"]),
    ("a", ["a"]),
    ("nnn", ["n", "n", "n"]),
])
def test_syllabify(word: str, expected: list[str]) -> None:
    assert syllabify(word) == expected


# ---- Latin -> UCSUR ----

def test_known_words_map_to_glyphs() -> None:
    assert latin_to_ucsur("mi olin e sina") == " ".join(
        [_g("mi"), _g("olin"), _g("e"), _g("sina")]
    )


def test_period_maps_to_middle_dot() -> None:
    assert latin_to_ucsur("mi pona.").endswith(chr(MIDDLE_DOT))


def test_question_mark_maps_to_middle_dot() -> None:
    assert latin_to_ucsur("toki?").endswith(chr(MIDDLE_DOT))


def test_exclamation_maps_to_middle_dot() -> None:
    assert latin_to_ucsur("pona!").endswith(chr(MIDDLE_DOT))


def test_colon_maps_to_middle_colon() -> None:
    assert latin_to_ucsur("toki:").endswith(chr(MIDDLE_COLON))


def test_comma_is_dropped() -> None:
    assert "," not in latin_to_ucsur("mi, sina")


def test_sentence_initial_capital_known_word_stays_a_word() -> None:
    # "Mi" at start of sentence is still the mi glyph, not a cartouche.
    out = latin_to_ucsur("Mi pona")
    assert chr(CARTOUCHE_START) not in out
    assert out.startswith(_g("mi"))


def test_capitalized_name_becomes_cartouche_of_syllable_glyphs() -> None:
    # "Mali" syllabifies to ["ma", "li"]; both syllables are TP words, so
    # each maps to its own glyph inside a cartouche.
    out = latin_to_ucsur("jan Mali li pona")
    expected = (
        _g("jan") + " "
        + chr(CARTOUCHE_START) + _g("ma") + _g("li") + chr(CARTOUCHE_END)
        + " " + _g("li") + " " + _g("pona")
    )
    assert out == expected


def test_cvn_syllable_with_no_word_falls_back_to_cv_plus_n() -> None:
    # "Sonja" syllabifies to ["son", "ja"]; "son" has no representative word,
    # so the encoder falls back to "so" + "n" (both have glyphs).
    out = latin_to_ucsur("Sonja")
    assert out.startswith(chr(CARTOUCHE_START))
    assert out.endswith(chr(CARTOUCHE_END))
    inner = out[1:-1]
    assert inner == _g("sona") + _g("n") + _g("jaki")


def test_unknown_lowercase_word_is_dropped() -> None:
    # The translator is for clean TP input; foreign lowercase falls through.
    assert latin_to_ucsur("hello") == ""


# ---- UCSUR -> Latin ----

def test_word_glyphs_to_latin() -> None:
    s = _g("mi") + _g("olin") + _g("e") + _g("sina")
    assert ucsur_to_latin(s) == "mi olin e sina"


def test_middle_dot_to_period() -> None:
    s = _g("mi") + _g("pona") + chr(MIDDLE_DOT)
    assert ucsur_to_latin(s) == "mi pona."


def test_middle_colon_to_colon() -> None:
    s = _g("toki") + chr(MIDDLE_COLON)
    assert ucsur_to_latin(s) == "toki:"


def test_cartouche_to_capitalized_name() -> None:
    s = chr(CARTOUCHE_START) + _g("ma") + _g("li") + chr(CARTOUCHE_END)
    assert ucsur_to_latin(s) == "Mali"


def test_extended_cartouche_behaves_like_normal_cartouche() -> None:
    s = chr(CARTOUCHE_EXT_START) + _g("ma") + _g("li") + chr(CARTOUCHE_EXT_END)
    assert ucsur_to_latin(s) == "Mali"


def test_cartouche_uses_first_syllable_of_multisyllable_glyph() -> None:
    # The "toki" glyph inside a cartouche contributes "to", not "toki".
    s = chr(CARTOUCHE_START) + _g("toki") + _g("ma") + chr(CARTOUCHE_END)
    assert ucsur_to_latin(s) == "Toma"


def test_stacking_joiner_yields_space_separated_words() -> None:
    s = _g("tomo") + chr(STACKING_JOINER) + _g("telo")
    assert ucsur_to_latin(s) == "tomo telo"


def test_scaling_joiner_yields_space_separated_words() -> None:
    s = _g("kasi") + chr(SCALING_JOINER) + _g("suwi")
    assert ucsur_to_latin(s) == "kasi suwi"


def test_long_glyph_extension_is_silently_stripped() -> None:
    s = _g("lawa") + chr(LONG_GLYPH_EXTENSION) + _g("ma")
    assert ucsur_to_latin(s) == "lawa ma"


def test_reverse_long_glyph_markers_are_silently_stripped() -> None:
    s = (
        chr(START_REVERSE_LONG_GLYPH) + _g("lawa")
        + chr(END_REVERSE_LONG_GLYPH) + _g("ma")
    )
    assert ucsur_to_latin(s) == "lawa ma"


def test_word_glyphs_without_separating_spaces_still_parse() -> None:
    # UCSUR codepoints are self-delimiting; no whitespace required between them.
    s = _g("mi") + _g("pona")
    assert ucsur_to_latin(s) == "mi pona"


# ---- round trips ----

@pytest.mark.parametrize("latin", [
    "mi pona.",
    "toki:",
    "mi olin e sina.",
    "jan Mali li pona.",
    "soweli suli li lon ma.",
    "Sonja li toki.",
])
def test_round_trip_latin_to_ucsur_to_latin(latin: str) -> None:
    assert ucsur_to_latin(latin_to_ucsur(latin)) == latin
