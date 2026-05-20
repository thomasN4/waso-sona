"""SitelenTokenizer: Latin-script Toki Pona ↔ sitelen pona UCSUR glyph tokens."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .glyphs import (
    ASCII_PUNCT_MAP, CARTOUCHE_END, CARTOUCHE_START, CODEPOINT_TO_WORD,
    KU_SULI_WORDS, MIDDLE_COLON, MIDDLE_DOT, PU_WORDS, PUNCTUATION_CODEPOINTS,
    WORD_TO_CODEPOINT,
)
from .syllabify import syllabify

# Special token strings (not in UCSUR; reserved at low IDs).
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# When syllabifying a name or unknown word, each syllable is rendered as the
# glyph of a Toki Pona word whose first letter matches the syllable's first
# letter. We use the first alphabetically-listed pu word for each letter so
# the choice is deterministic and round-trippable.
_LETTER_TO_GLYPH_WORD = {
    "a": "a", "e": "e", "i": "ijo", "o": "o", "u": "unpa",
    "p": "pakala", "t": "tan", "k": "kala", "s": "sama",
    "m": "ma", "n": "nanpa", "l": "la", "j": "jaki", "w": "walo",
}

# Word-splitter: a run of letters, OR a single punctuation char we care about.
_TOKEN_RE = re.compile(r"[A-Za-z]+|[.!?:,]")


class SitelenTokenizer:
    """Word-level tokenizer that maps Latin-script Toki Pona to UCSUR glyphs.

    Vocabulary layout (token IDs):
        0..3            : special tokens (<pad>, <unk>, <bos>, <eos>)
        4..n_words+3    : pu + ku suli word glyphs
        next            : structural punctuation (cartouches, period, colon)
    """

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)
        # Word glyphs (each token is its single-character glyph string).
        for word in PU_WORDS + KU_SULI_WORDS:
            self.id_to_token.append(chr(WORD_TO_CODEPOINT[word]))
        # Structural punctuation.
        for cp in sorted(PUNCTUATION_CODEPOINTS):
            self.id_to_token.append(chr(cp))

        self.token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self.id_to_token)
        }

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]

    # ---- introspection ----
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    # ---- core API ----
    def tokenize(self, text: str) -> list[str]:
        """Return the list of glyph (and special) tokens for `text`."""
        out: list[str] = []
        for match in _TOKEN_RE.finditer(text):
            piece = match.group(0)
            if piece in ASCII_PUNCT_MAP:
                out.append(chr(ASCII_PUNCT_MAP[piece]))
            elif piece == ",":
                # Toki Pona writing typically omits commas; drop them.
                continue
            else:
                out.extend(self._tokenize_word(piece))
        return out

    def _tokenize_word(self, word: str) -> list[str]:
        lower = word.lower()
        is_name = word[0].isupper() and not lower in WORD_TO_CODEPOINT
        if not is_name and lower in WORD_TO_CODEPOINT:
            return [chr(WORD_TO_CODEPOINT[lower])]
        # Unknown or capitalized -> cartouche of syllable-initial glyphs.
        glyphs = [chr(CARTOUCHE_START)]
        for syl in syllabify(lower):
            first = syl[0]
            target = _LETTER_TO_GLYPH_WORD.get(first)
            if target is None:
                # Non-TP letter: emit <unk> and continue.
                glyphs.append(self.unk_token)
            else:
                glyphs.append(chr(WORD_TO_CODEPOINT[target]))
        glyphs.append(chr(CARTOUCHE_END))
        return glyphs

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for tok in self.tokenize(text):
            ids.append(self.token_to_id.get(tok, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Render token IDs as a sitelen pona UCSUR string.

        Word glyphs are separated by spaces (matching common sitelen pona
        rendering). Inside cartouches, syllable glyphs are joined directly.
        """
        specials = {
            self.pad_token_id, self.unk_token_id,
            self.bos_token_id, self.eos_token_id,
        }
        pieces: list[str] = []
        in_cartouche = False
        for tid in ids:
            if skip_special_tokens and tid in specials:
                continue
            tok = self.id_to_token[tid]
            if tok == chr(CARTOUCHE_START):
                pieces.append(" " + tok)
                in_cartouche = True
            elif tok == chr(CARTOUCHE_END):
                pieces.append(tok)
                in_cartouche = False
            elif in_cartouche:
                pieces.append(tok)
            else:
                pieces.append(" " + tok)
        return "".join(pieces).strip()

    # ---- HuggingFace-style persistence ----
    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "vocab.json").write_text(
            json.dumps(self.token_to_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "SitelenTokenizer",
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "SitelenTokenizer":
        # The vocabulary is deterministic, so loading just re-instantiates.
        # We still validate that the saved vocab matches.
        path = Path(load_directory)
        tokenizer = cls()
        vocab_file = path / "vocab.json"
        if vocab_file.exists():
            saved = json.loads(vocab_file.read_text(encoding="utf-8"))
            if saved != tokenizer.token_to_id:
                raise ValueError(
                    "Saved vocab does not match current SitelenTokenizer "
                    "vocabulary (UCSUR mapping changed?)."
                )
        return tokenizer
