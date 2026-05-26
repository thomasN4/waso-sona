# waso-sona

A talking bird that lives in your desktop.

## Direction

Working toward a small language model that understands Toki Pona written in
[sitelen pona](https://en.wikipedia.org/wiki/Sitelen_Pona) via the
[UCSUR](https://www.kreativekorp.com/ucsur/charts/sitelen.html) code points,
trained locally on consumer hardware (developed on an RTX 5060, 8 GB VRAM).

Current pieces:

- `sitelen/` — bidirectional translator between Latin Toki Pona and
  sitelen pona UCSUR (syllable-aware). UCSUR is handled entirely at the
  application layer; the model itself trains and runs on Latin script.
- `scripts/fetch_data.py` — pulls Toki Pona corpora into `data/raw/`.

## Setup

Requires Python 3.10–3.13 and (for GPU training) a CUDA 12.8-capable NVIDIA
GPU. The PyTorch wheels are pulled from the `cu128` index, configured in
`pyproject.toml`.

```sh
uv venv --python 3.13
uv pip install -e .
```

Run the tests with `uv run pytest`.
