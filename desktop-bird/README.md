# desktop-bird

A desktop-pet bird that lives on the desktop, perches on the active window, and
flies around. Wayland-only, targeting **KDE Plasma 6 (KWin)** and **COSMIC DE
(cosmic-comp)**. See [`RESEARCH.md`](RESEARCH.md) for the architecture and the
Wayland-protocol reasoning behind it.

## Status

- **Slice 1 — shared renderer (done).** A transparent, click-through
  `wlr-layer-shell` surface fills the output; the bird is drawn at an offset
  inside it. Portable to any layer-shell compositor.
- **Slice 2 — COSMIC window tracker (done).** On COSMIC, the bird reads the
  active window's geometry from `cosmic-toplevel-info` and perches on its top
  edge; off COSMIC it falls back to wandering. KWin tracking is next.

## Licensing

The renderer depends only on the MIT Smithay crates. The **COSMIC backend**
pulls in `cosmic-client-toolkit` / `cosmic-protocols`, which are **GPL-3.0-only**,
so the crate as a whole is currently **GPL-3.0-only** (see `Cargo.toml`). If an
MIT/permissive build matters, the COSMIC backend would need to be put behind an
optional cargo feature (not done yet).

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
| `src/tracker.rs` | `WindowTracker` trait + `WindowInfo` — the per-compositor abstraction. |
| `src/cosmic.rs` | `CosmicTracker` — active-window geometry via cctk `ToplevelInfoState`. |

## Window tracking (COSMIC)

`cosmic-toplevel-info` is dispatched on the **same** Wayland event queue as the
renderer (its globals live on the same registry), so no extra event loop is
needed yet. At startup we check the registry for `zcosmic_toplevel_info_v1`; if
present we bind cctk's `ToplevelInfoState`, otherwise tracking is disabled. The
active (`Activated`) toplevel's geometry drives a simple perch target on its top
edge. The active window is logged to stderr on change.

> The KWin backend (a `.kwinscript` pushing geometry over **D-Bus**) *will* need
> a second event source — that's when the loop moves to `calloop`.

## How it renders

One output-sized `Argb8888` (premultiplied BGRA) layer surface on the `Top`
layer, anchored to all four edges with `exclusive_zone = -1`. The bird is painted
at an `(x, y)` offset; each frame the buffer is cleared (so undamaged regions are
correctly transparent) and only the bird's old+new bounding boxes are damaged, so
the compositor re-uploads just a small region. An empty `wl_region` input region
makes the whole surface click-through.

## Roadmap (next slices)

- **KWin `WindowTracker` backend**: ship a `.kwinscript` that pushes
  `frameGeometry` over D-Bus (the COSMIC backend already exists). Needs `calloop`.
- Real **`BirdBrain`** state machine (fly-to / perch / wander) driven by tracker
  geometry, replacing the interim perch logic in `app.rs`.
- Multi-output (per-output surfaces + coordinate translation).
- Optional draggable/pettable bird (sprite-shaped, non-empty input region).

> **Note for the tracker slice:** adding a second event source (a second Wayland
> queue for the COSMIC protocol, or a D-Bus connection for the KWin script) will
> want a `calloop` event loop (`calloop` + `calloop-wayland-source`, both already
> pulled in transitively) instead of the current `blocking_dispatch` loop — so we
> don't rebuild the loop twice.
