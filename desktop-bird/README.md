# desktop-bird

A desktop-pet bird that lives on the desktop, perches on the active window, and
flies around. Wayland-only, targeting **KDE Plasma 6 (KWin)** and **COSMIC DE
(cosmic-comp)**. See [`RESEARCH.md`](RESEARCH.md) for the architecture and the
Wayland-protocol reasoning behind it.

## Status

**Slice 1 — the shared renderer (done).** A transparent, click-through
`wlr-layer-shell` surface fills the output; the bird is drawn at an offset inside
it and wanders semi-randomly. No window tracking yet — the bird does not perch on
real windows. That part is per-compositor and comes next (see *Roadmap*).

## Build & run

```sh
cargo run --release
```

Must be run inside a Wayland session whose compositor implements
`wlr-layer-shell` (KWin / cosmic-comp do; GNOME/Mutter does not). You should see
a small bird drift around the screen; clicks pass straight through it to the
windows underneath, and it never takes keyboard focus.

### Custom sprite (optional)

By default a procedurally generated placeholder bird is used. To use real art,
point at a directory of PNG animation frames (sorted by filename, all the same
size; the bird should face **right**):

```sh
BIRD_SPRITE_DIR=/path/to/frames cargo run --release
```

## Layout

| File | Responsibility |
|------|----------------|
| `src/main.rs` | Wayland bring-up: bind globals, create the layer surface, set anchors / empty input region, run the dispatch loop. |
| `src/app.rs` | `AppState` + SCTK delegate impls; per-frame tick and damage-tracked draw. |
| `src/render.rs` | `Rect` math and the RGBA→premultiplied-BGRA sprite blit. |
| `src/sprite.rs` | `Sprite`/`Frame`, the placeholder bird, and the PNG-frame loader. |
| `src/motion.rs` | `Wander` — placeholder motion (stand-in for the future shared `BirdBrain`). |

## How it renders

One output-sized `Argb8888` (premultiplied BGRA) layer surface on the `Top`
layer, anchored to all four edges with `exclusive_zone = -1`. The bird is painted
at an `(x, y)` offset; each frame the buffer is cleared (so undamaged regions are
correctly transparent) and only the bird's old+new bounding boxes are damaged, so
the compositor re-uploads just a small region. An empty `wl_region` input region
makes the whole surface click-through.

## Roadmap (next slices)

- **`WindowTracker` trait** to get the active window's geometry, with two backends:
  - **COSMIC**: bind `cosmic-toplevel-info-unstable-v1` and read its `geometry` event.
  - **KWin**: ship a `.kwinscript` that pushes `frameGeometry` over D-Bus.
- Real **`BirdBrain`** state machine (fly-to / perch / wander) driven by tracker geometry.
- Multi-output (per-output surfaces + coordinate translation).
- Optional draggable/pettable bird (sprite-shaped, non-empty input region).

> **Note for the tracker slice:** adding a second event source (a second Wayland
> queue for the COSMIC protocol, or a D-Bus connection for the KWin script) will
> want a `calloop` event loop (`calloop` + `calloop-wayland-source`, both already
> pulled in transitively) instead of the current `blocking_dispatch` loop — so we
> don't rebuild the loop twice.
