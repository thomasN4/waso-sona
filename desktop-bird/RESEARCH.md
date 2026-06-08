# Desktop bird — Wayland research (Plasma 6 + COSMIC)

Goal: a sprite bird that lives on the desktop, perches on the active window, and
flies around semi-randomly. Wayland-only, must work on **KDE Plasma 6 (KWin)** and
**COSMIC DE (cosmic-comp)**.

This doc records what's actually possible on Wayland and where the two compositors
diverge, so we can pick an architecture before writing code.

---

## TL;DR

A desktop pet needs three capabilities. Two are easy and portable; one is the whole
ballgame.

| Capability | Mechanism | Plasma 6 | COSMIC | Portable? |
|---|---|---|---|---|
| Always-on-top transparent overlay surface | `wlr-layer-shell` (`zwlr_layer_shell_v1`) | ✅ | ✅ | ✅ yes |
| Place sprite at arbitrary x,y | layer-shell anchor-corner + margins trick | ✅ | ✅ | ✅ yes |
| **Know the active window's geometry (to perch)** | **compositor-specific** | KWin script → D-Bus | `cosmic-toplevel-info` geometry event | ❌ **no — needs a backend per DE** |

The fly-around / overlay rendering is solved and identical on both. The *perching*
feature forces a small per-compositor abstraction because **Wayland deliberately
refuses to tell a normal client where any window (including other apps') is**.

---

## 1. The compositors

Both are pure-Wayland, no X11 baseline assumptions needed.

- **cosmic-comp** (COSMIC) is a from-scratch compositor built on **smithay** (Rust).
  It implements 30+ Wayland protocol objects including **xdg-shell, layer-shell, and
  workspace management**, plus XWayland for legacy apps. Applets themselves use
  layer-shell, so it's a first-class, well-exercised path.
- **KWin** (Plasma 6) is a mature compositor with full **layer-shell** support
  (used by Plasma panels, krunner, notifications, lock screen).

So `zwlr_layer_shell_v1` is available on both → that's our rendering surface.

> ⚠️ GNOME/Mutter does **not** implement layer-shell and has no plans to. If we ever
> want GNOME we'd need an entirely different (much worse) approach. Out of scope —
> we're targeting Plasma + COSMIC only, which is the right call.

---

## 2. Rendering the bird — layer-shell (portable)

`wlr-layer-shell-unstable-v1` lets a client render into one of four global layers,
bottom-to-top: **`background`, `bottom`, `top`, `overlay`**.

- For a pet that sits on top of windows, use the **`top`** layer (above normal
  windows, below the system overlay) or **`overlay`** (above everything incl.
  fullscreen). Likely a setting.
- Transparency: layer surfaces support an alpha channel like any wl_surface — draw
  the sprite with a transparent background.
- **Click-through**: set an *empty* `wl_surface` input region
  (`wl_surface.set_input_region` with an empty region) so clicks pass to the window
  underneath. If we want the bird to be **draggable/pettable**, set the input region
  to the sprite's bounding box instead, and empty everywhere else. Trade-off to
  decide later.
- **Keyboard**: set keyboard-interactivity to `none` so the bird never steals focus.

### Positioning gotcha (important)

Layer-shell surfaces **cannot be placed at an arbitrary (x, y)**. You only get:
- **anchor**: a bitfield of edges (`top`/`bottom`/`left`/`right`). Two orthogonal
  edges = a corner; one edge = centered on it; none = centered on output.
- **margin**: distance from the anchored edge(s).

**The trick** (standard for notification daemons / floating widgets): anchor to
`top | left` (the output's top-left corner), size the surface to the sprite, and set
`margin_top = y`, `margin_left = x`. That gives effectively free positioning within
an output. Margins are int32, large values are fine.

Consequences:
- Coordinates are **per-output**. On multi-monitor we create a layer surface per
  output (or per the output the bird is currently on) and translate global coords →
  output-local.
- The bird's stacking is **whole-layer**, not per-window. It always renders above (or
  below) *all* windows in its layer. So the bird can sit on the visual top edge of a
  window, but it can't be sandwiched between window A and window B. For "perch on the
  titlebar of the active window," `top`/`overlay` layer looks correct — the bird
  visibly rests on the window's top edge.

A real reference: **koverlay** (Qt6 + QML + LayerShellQt) — click-through,
always-on-top overlay pinned via layer-shell. Confirms the approach is viable today.

---

## 3. The hard part — active-window geometry (NOT portable)

To perch, fly to, and react to windows, the bird needs the active window's
`{x, y, width, height}` in screen coordinates, updated as it moves/resizes.

**Wayland forbids this for normal clients by design.** A client cannot know its own
absolute position, let alone another app's. Quote from the design rationale: *"It is
a design decision in Wayland to not expose absolute window positions to clients at
all... you can only know which outputs it overlaps with."* Motivations: compositor
flexibility (VR, mirroring) and security (no overlaying a fake password box). This is
the single biggest difference from X11, where any client can inspect/position any
window.

The base `ext-foreign-toplevel-list-v1` protocol (supported by both KWin and COSMIC)
gives us a **list of toplevels with title, app_id, identifier, and via the handle the
activated/maximized/minimized/fullscreen state** — but **intentionally NO geometry**.
Good enough to know *which* window is active and its title; not enough to perch.

So geometry must come from a compositor-specific channel:

### 3a. COSMIC — native protocol ✅ (clean)

`cosmic-toplevel-info-unstable-v1` extends the foreign-toplevel system and adds a
**`geometry` event**:
- params: `output, x, y, width, height` (upper-left corner + size, ints).
- fires *once on creation per entered output*, and *again whenever geometry changes
  relative to any output*.
- plus title, app_id, and a state array including `activated`.

This is exactly what we need, delivered straight to our client as standard Wayland
events. No helper process. (Accessible from Rust via `cosmic-client-toolkit` / `cctk`,
or by binding the protocol XML directly with `wayland-client`.)

### 3b. KWin / Plasma 6 — KWin script bridge ⚠️ (works, more moving parts)

KWin exposes **no client protocol** that hands window geometry to an arbitrary app.
The supported route is a **KWin script** that runs *inside* the compositor (QJSEngine
sandbox) where it has the scripting API, and pushes data out:

- `workspace.activeWindow` → current active window object.
- `window.frameGeometry` → `{x, y, width, height}` in global logical coords.
- Signals: `workspace.windowActivated(window)` (fires on focus change, NULL if none),
  and per-window `frameGeometryChanged` / move-resize signals.
- The script ships geometry to our app via **`callDBus(...)`** (async D-Bus call to a
  service our app registers). Could also write to a unix socket/file, but D-Bus is the
  idiomatic path.

Proven references implementing exactly this pattern:
- **c-massie/FocusNotifier** — KWin script + systemd service + CLI exposing the focused
  window's info, explicitly for Wayland *and* X11.
- **bouteillerAlan/window_signal** — KWin script that emits a D-Bus signal on focus
  change.

Packaging implication: on Plasma we must **install + enable a `.kwinscript`** as part
of setup (and reload KWin scripting). Slightly more friction than COSMIC, but standard.

> Note: `frameGeometryChanged` may be throttled / only fire at the end of an
> interactive resize depending on mode. For a pet that's fine — we don't need
> sub-frame precision.

---

## 4. Recommended architecture

A single app (likely **Rust** — matches the Wayland/smithay ecosystem and gives us
`smithay-client-toolkit` for layer-shell + `cctk` for the COSMIC protocol) with a
thin backend trait for the one divergent capability:

```
WindowTracker (trait)
  ├─ events: active_window_changed{geometry, title, app_id}, window_moved, window_closed
  ├─ CosmicTracker   → binds cosmic-toplevel-info-v1, listens to `geometry`
  └─ KWinTracker     → registers a D-Bus service, ships+loads a .kwinscript,
                       receives geometry over D-Bus

Renderer (shared)
  └─ zwlr_layer_shell_v1 surface(s), one per output
       anchor=top|left, margins=(x,y), empty/region input, `top` or `overlay` layer
       draws sprite frames; transparent; no keyboard focus

BirdBrain (shared)
  └─ state machine: idle / fly-to / perch / wander; reads tracker geometry,
     drives renderer margins each frame (semi-random walk + perch targeting)
```

Detection of which backend to use: check for the COSMIC toplevel-info global in the
registry; else assume KWin and use the script bridge. (Could also sniff
`XDG_CURRENT_DESKTOP`.)

### Toolkit options considered
- **Rust + smithay-client-toolkit + cctk** — best fit, raw control, native COSMIC
  protocol support. Recommended.
- **gtk4 + gtk4-layer-shell** — fast UI/animation, but COSMIC geometry protocol needs
  hand-rolled bindings anyway; heavier dep.
- **Qt6 + LayerShellQt** (the koverlay/Shijima-Qt stack) — proven, but Qt dep and we'd
  still hand-roll the COSMIC + KWin geometry bridges.

### Reality check from prior art
- **Shijima-Qt** (Shimeji runner) added an alpha Wayland backend that *requires*
  layer-shell — matches our plan — but the project is archived/unmaintained.
- **estenv/linux-shimeji** stayed X11-only, author calling the Wayland situation
  "grim" (pre-foreign-toplevel/COSMIC-geometry era). The geometry protocols above are
  what make our approach more tractable now than when they gave up.

---

## 5. Decisions

Locked in:

1. **Language/toolkit — Rust + smithay-client-toolkit** (+ `cctk` for the COSMIC
   protocol). ✅ decided.
2. **Perch target — active window only.** The bird perches on / flies to the currently
   active window's top edge; we do not track all toplevels for now. This simplifies
   both backends: COSMIC needs only the `activated` toplevel's `geometry`; the KWin
   script only watches `workspace.activeWindow` + `windowActivated`. (Hopping between
   all visible windows is a possible later extension — both backends can supply the
   data.) ✅ decided.

Still open (decide as we build):

3. **Draggable bird?** — decides whether input region is empty (pure click-through) or
   sprite-shaped.
4. **Layer** — `top` (under fullscreen/system overlays) vs `overlay` (above all).
5. **Multi-monitor** — per-output layer surfaces + coordinate translation (needed,
   just confirming scope).
6. **KWin packaging** — how we install/enable the `.kwinscript` and register the D-Bus
   service cleanly (and uninstall).
7. Where the bird sprite assets / animation format come from (could tie into the
   project's `sitelen` art? — TBD).

---

## Sources
- [pop-os/cosmic-comp — DeepWiki](https://deepwiki.com/pop-os/cosmic-comp) · [Wayland Integration](https://deepwiki.com/pop-os/cosmic-comp/7-wayland-integration)
- [COSMIC — ArchWiki](https://wiki.archlinux.org/title/COSMIC)
- [COSMIC toplevel info protocol — Wayland Explorer](https://wayland.app/protocols/cosmic-toplevel-info-unstable-v1)
- [wlr-layer-shell-unstable-v1 — Wayland Explorer](https://wayland.app/protocols/wlr-layer-shell-unstable-v1)
- [ext-foreign-toplevel-list-v1 — Wayland Explorer](https://wayland.app/protocols/ext-foreign-toplevel-list-v1)
- [Wayland's Never-Ending Opposition To Multi-Window Positioning — Hackaday](https://hackaday.com/2025/11/11/waylands-never-ending-opposition-to-multi-window-positioning/)
- [Window positions under Wayland — Mir docs](https://canonical.com/mir/docs/stable/explanation/window-positions-under-wayland/)
- [KWin scripting API — KDE Developer](https://develop.kde.org/docs/plasma/kwin/api/)
- [Geometry handling in KWin/Wayland — Vlad Zahorodnii](https://blog.vladzahorodnii.com/2022/01/15/geometry-handling-in-kwin-wayland/)
- [c-massie/FocusNotifier](https://github.com/c-massie/FocusNotifier) · [bouteillerAlan/window_signal](https://github.com/bouteillerAlan/window_signal)
- [koverlay](https://github.com/erik96/koverlay) · [pixelomer/Shijima-Qt](https://github.com/pixelomer/Shijima-Qt) · [estenv/linux-shimeji](https://github.com/estenv/linux-shimeji)
