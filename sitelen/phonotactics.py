"""
Phonotactic rules (pu-canon):
1. Only letters p t k s m n l j w (consonants) and a e i o u (vowels).
2. Every syllable is (C)V(n): optional onset consonant, one vowel, optional coda n.
3. Only the first syllable may lack an onset — no vowel directly following a vowel ("ijo" ✓, "aula" ✗).
4. Forbidden syllables: ti, ji, wo, wu (with or without coda n — "tin" is also illegal).
5. A coda n may not be followed by n or m ("nanpa" ✓, "panma" ✗).
6. Special case: the bare word "n" is legal — it's an attested word ("hmm") and exists in the UCSUR vocab, even though it breaks rule 2.
"""

from .syllabify import (syllabify, CONSONANTS, VOWELS)

FORBIDDEN_SYLLABLES = {"ti", "ji", "wo", "wu"}
LETTERS = CONSONANTS + VOWELS

def is_legal_word(word: str) -> bool:
    """True if `word` is phonotactically legal Toki Pona (shape, not dictionary)."""
    if word == "":
        # b/c we expect that syllabify("") == []
        return False
    
    syllables = syllabify(word)
    i = 0
    prev = ""
    
    if syllables == ["n"]:
        # rule 6
        return True
    
    for syl in syllables:
        if not _has_only_legal_letters(syl):
            # rule 1
            return False
        if i > 0 and prev[-1] in VOWELS and syl[0] in VOWELS:
            # rule 3
            return False
        if len(syl) > 1:
            if syl[:2] in FORBIDDEN_SYLLABLES:
                # rule 4
                return False
            if i > 0 and prev[-1] == "n" and syl[0] in "nm":
                # rule 5
                # Note: due to how syllabify() handles "n", syl[0] == "n" never happens,
                # and it's the rule 2 branch of the loop which handles this case.
                return False
        else:
            if syl not in VOWELS:
                # this with the rest should handle rule 2
                return False
        prev = syl
        i += 1
    
    return True

def _has_only_legal_letters(syl: str) -> bool:
    return all([c in LETTERS for c in syl])