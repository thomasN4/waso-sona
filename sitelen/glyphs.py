"""UCSUR codepoint allocations for sitelen pona (pu + ku suli)."""

PU_WORDS = [
    "a", "akesi", "ala", "alasa", "ale", "anpa", "ante", "anu", "awen", "e",
    "en", "esun", "ijo", "ike", "ilo", "insa", "jaki", "jan", "jelo", "jo",
    "kala", "kalama", "kama", "kasi", "ken", "kepeken", "kili", "kiwen", "ko",
    "kon", "kule", "kulupu", "kute", "la", "lape", "laso", "lawa", "len",
    "lete", "li", "lili", "linja", "lipu", "loje", "lon", "luka", "lukin",
    "lupa", "ma", "mama", "mani", "meli", "mi", "mije", "moku", "moli",
    "monsi", "mu", "mun", "musi", "mute", "nanpa", "nasa", "nasin", "nena",
    "ni", "nimi", "noka", "o", "olin", "ona", "open", "pakala", "pali",
    "palisa", "pan", "pana", "pi", "pilin", "pimeja", "pini", "pipi", "poka",
    "poki", "pona", "pu", "sama", "seli", "selo", "seme", "sewi", "sijelo",
    "sike", "sin", "sina", "sinpin", "sitelen", "sona", "soweli", "suli",
    "suno", "supa", "suwi", "tan", "taso", "tawa", "telo", "tenpo", "toki",
    "tomo", "tu", "unpa", "uta", "utala", "walo", "wan", "waso", "wawa",
    "weka", "wile",
]

KU_SULI_WORDS = [
    "namako", "kin", "oko", "kipisi", "leko", "monsuta", "tonsi", "jasima",
    "kijetesantakalu", "soko", "meso", "epiku", "kokosila", "lanpan", "n",
    "misikeke", "ku",
]

UCSUR_BASE = 0xF1900

WORD_TO_CODEPOINT: dict[str, int] = {}
for _i, _word in enumerate(PU_WORDS + KU_SULI_WORDS):
    WORD_TO_CODEPOINT[_word] = UCSUR_BASE + _i

CODEPOINT_TO_WORD: dict[int, str] = {v: k for k, v in WORD_TO_CODEPOINT.items()}

# Structural punctuation (UCSUR)
CARTOUCHE_START = 0xF1990
CARTOUCHE_END = 0xF1991
CARTOUCHE_EXT_START = 0xF1992
CARTOUCHE_EXT_END = 0xF1993
LONG_GLYPH_EXTENSION = 0xF1994
STACKING_JOINER = 0xF1995
SCALING_JOINER = 0xF1996
START_REVERSE_LONG_GLYPH = 0xF1997
END_REVERSE_LONG_GLYPH = 0xF1998
MIDDLE_DOT = 0xF199C   # "."
MIDDLE_COLON = 0xF199D  # ":"

PUNCTUATION_CODEPOINTS = {
    CARTOUCHE_START, CARTOUCHE_END,
    CARTOUCHE_EXT_START, CARTOUCHE_EXT_END,
    LONG_GLYPH_EXTENSION, STACKING_JOINER, SCALING_JOINER,
    START_REVERSE_LONG_GLYPH, END_REVERSE_LONG_GLYPH,
    MIDDLE_DOT, MIDDLE_COLON,
}

# Mapping from ASCII punctuation to UCSUR codepoints
ASCII_PUNCT_MAP = {
    ".": MIDDLE_DOT,
    "!": MIDDLE_DOT,
    "?": MIDDLE_DOT,
    ":": MIDDLE_COLON,
}
