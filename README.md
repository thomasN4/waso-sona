# waso-sona

A small Toki Pona language model — and a desktop pet bird that renders and
*speaks* it. (*waso sona* ≈ "knowing bird".)

The bird lives on a transparent overlay on your desktop, hops around, perches on
whatever window you're using, and pipes up now and then in
[sitelen pona](https://en.wikipedia.org/wiki/Sitelen_Pona) — its speech generated
locally by a tiny language model trained from scratch on Toki Pona.

## What it is

Two halves of one project:

### 1. The model

A small Toki Pona language model trained locally on consumer hardware (developed
on an RTX 5060, 8 GB VRAM). It's built with a teacher→student pipeline:

- **Teacher** — a QLoRA fine-tune of Gemma 4 on real Toki Pona corpora.
- **Augment** — the teacher (plus targeted synthesis) expands and rebalances the
  training corpus.
- **Student** — a tiny from-scratch language model trained on that corpus, then
  given a cheerful bird persona via a small SFT pass so it speaks as the desktop
  bird.

The model operates on **Latin-script** Toki Pona; conversion to and from sitelen
pona [UCSUR](https://www.kreativekorp.com/ucsur/charts/sitelen.html) code points is
an application-layer concern, handled by the translator in `sitelen/`.

See [`PIPELINE.md`](PIPELINE.md) for the full design, data sources, and current
training status.

### 2. The desktop bird

A Rust + Wayland application in [`desktop-bird/`](desktop-bird/) that renders the
bird on a click-through layer-shell overlay, drives its behaviour with a small
state machine (wander → approach → perch → flit), and shows sitelen pona speech
bubbles fed by the model. It perches on the active window by tracking it through
the compositor:

- **KWin / Plasma 6** — via a KWin script that reports window geometry over D-Bus.
- **COSMIC** — via the `cosmic-toplevel-info` protocol.
- Otherwise it falls back to free wandering.

See [`desktop-bird/README.md`](desktop-bird/README.md) for the app's design and the
Wayland integration details.

## Repo layout

| Path | What's there |
|------|--------------|
| `sitelen/` | Bidirectional Latin Toki Pona ↔ sitelen pona UCSUR translator (syllable-aware). |
| `scripts/` | The model pipeline — fetch/filter corpora, train the teacher, augment, train the student, and `talk_to_bird.py` / `bird_persona.py` for the bird's voice. |
| `desktop-bird/` | The Rust/Wayland desktop pet app (renderer, behaviour, speech bubbles, window trackers). |
| `data/` | Raw and processed training corpora. |
| `models/` | Trained checkpoints and tokenizers. |
| `PIPELINE.md` | Authoritative design + status doc for the model pipeline. |

## Setup

Requires Python 3.10–3.13 and, for GPU training, a CUDA 12.8-capable NVIDIA GPU.
The PyTorch wheels are pulled from the `cu128` index, configured in
`pyproject.toml`.

```sh
uv venv --python 3.13
uv pip install -e .
```

Run the tests with `uv run pytest`.

## Running the bird

The bird is a Wayland app: it needs a session with `wlr-layer-shell` support
(KWin 6 or cosmic-comp — GNOME/Mutter is not supported). From `desktop-bird/`,
pipe the model's output into the renderer:

```sh
../.venv/bin/python ../scripts/talk_to_bird.py --loop --ucsur | cargo run --release
```

To see the bird and its speech bubbles without the model, use the demo feed:

```sh
python demo.py | cargo run --release
```
