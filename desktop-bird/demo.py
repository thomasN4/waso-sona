#!/usr/bin/env python3
"""Feed hardcoded Toki Pona test sentences to desktop-bird as sitelen pona.

This is a stand-in for the real Toki Pona model. It reuses the project's
application-layer translator (`sitelen.latin_to_ucsur`) to turn Latin-script
Toki Pona into sitelen pona (UCSUR), and prints one bubble per line on stdout in
the bird's stdin protocol::

    <ucsur text>\\t<seconds>

It paces output so each bubble shows before the next. The bird is a thin display
client — it just renders whatever UCSUR text it is handed — so when the model
lands it pipes its `latin_to_ucsur` output into the bird exactly the same way.

Usage (from the repo root, inside a KWin / COSMIC Wayland session)::

    python desktop-bird/demo.py | cargo run --release --manifest-path desktop-bird/Cargo.toml

Stop with Ctrl-C.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

# Import the repo's `sitelen` package (one directory up from desktop-bird/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitelen import latin_to_ucsur  # noqa: E402

# Plain dictionary-word sentences only (no proper names): the bird renders one
# UCSUR glyph per word and does no OpenType shaping, so cartouches would not form.
SENTENCES = [
    "toki a",
    "mi moku e kili",
    "sina pona mute",
    "waso lili li tawa sewi",
    "tenpo suno ni li pona",
]

# Seconds each bubble stays up (must match the pacing so bubbles don't overlap).
SECONDS = 4.0


def main() -> int:
    try:
        for sentence in itertools.cycle(SENTENCES):
            ucsur = latin_to_ucsur(sentence)
            print(f"{ucsur}\t{SECONDS}", flush=True)
            time.sleep(SECONDS)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
