"""Tests for the shared corpus/translation gate `augment_corpus._filter_sentence`.

Focus: the phonotactics check on proper names. Capitalized tokens skip the
dictionary, but must be phonotactically legal TP word-shapes (`sitelen.is_legal_word`),
so leftover English names are rejected with reason `illegal_name`.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402

_filter_sentence = augment_corpus._filter_sentence


@pytest.mark.parametrize("sentence", [
    "jan Tan li pona lukin.",       # TP-ized name — legal shape
    "Toki li pona tawa mi.",        # sentence-initial capitalized real word
    "ma Mewika li suli.",           # multi-syllable transliterated name
])
def test_legal_names_accepted(sentence: str) -> None:
    ok, reason = _filter_sentence(sentence)
    assert ok, f"expected accept, got reject={reason!r}"


@pytest.mark.parametrize("sentence", [
    "mama Tom li tawa.",            # "Tom" → coda-less final "m", illegal
    "jan London li pona lukin.",    # "d" is not a TP letter
])
def test_illegal_names_rejected(sentence: str) -> None:
    assert _filter_sentence(sentence) == (False, "illegal_name")
