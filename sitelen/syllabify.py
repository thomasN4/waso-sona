"""Toki Pona syllabifier.

Phonotactics: (C)V(N) where
  C ∈ {p, t, k, s, m, n, l, j, w}
  V ∈ {a, e, i, o, u}
  N = 'n'

Disallowed onsets: ti, ji, wo, wu, nm, nn (but only the first three matter
for choosing where to split; the final-n / next-onset-n ambiguity is resolved
by checking whether 'n' would create a forbidden boundary).
"""
import re

CONSONANTS = "ptksmnljw"
VOWELS = "aeiou"

# Greedy syllable regex: optional onset + vowel + optional coda 'n'.
# Coda 'n' is only consumed if it is NOT followed by another vowel or 'n'
# (which would mean it actually belongs to the next syllable's onset, or
# forms a forbidden 'nn' cluster).
_SYLLABLE_RE = re.compile(
    rf"([{CONSONANTS}]?)([{VOWELS}])(n(?![{VOWELS}n]))?",
    re.IGNORECASE,
)


def syllabify(word: str) -> list[str]:
    """Split a Latin-script string into Toki Pona syllables.

    Non-matching characters are emitted as single-character "syllables" so
    the caller can decide how to handle them (e.g. drop or fall back).
    """
    word = word.lower()
    syllables: list[str] = []
    i = 0
    while i < len(word):
        m = _SYLLABLE_RE.match(word, i)
        if m and m.end() > m.start():
            syllables.append(m.group(0))
            i = m.end()
        else:
            syllables.append(word[i])
            i += 1
    return syllables
