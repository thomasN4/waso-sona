//! desktop-bird — shared Wayland layer-shell renderer.
//!
//! Puts a transparent, click-through bird on the desktop and flies it around
//! with a placeholder wander. No window tracking yet — that (perching on the
//! active window) is the next slice and is per-compositor (see RESEARCH.md).
//!
//! Works on any compositor that implements wlr-layer-shell — notably KWin
//! (Plasma 6) and cosmic-comp (COSMIC).

mod app;
mod motion;
mod render;
mod sprite;

use app::AppState;
use smithay_client_toolkit::{
    compositor::{CompositorState, Region},
    output::OutputState,
    registry::RegistryState,
    shell::{
        wlr_layer::{Anchor, KeyboardInteractivity, Layer, LayerShell},
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm},
};
use sprite::Sprite;
use wayland_client::{globals::registry_queue_init, Connection};

fn main() {
    let conn = Connection::connect_to_env()
        .expect("failed to connect to a Wayland compositor (is WAYLAND_DISPLAY set?)");

    let (globals, mut event_queue) =
        registry_queue_init(&conn).expect("failed to initialise the Wayland registry");
    let qh = event_queue.handle();

    let compositor = CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
    let layer_shell = LayerShell::bind(&globals, &qh)
        .expect("wlr-layer-shell is not available (need KWin, cosmic-comp, or another wlroots-style compositor)");
    let shm = Shm::bind(&globals, &qh).expect("wl_shm is not available");

    // One layer surface that fills the output. The compositor picks the output
    // (single-output for now); the configure event reports the real size.
    let surface = compositor.create_surface(&qh);
    let layer =
        layer_shell.create_layer_surface(&qh, surface, Layer::Top, Some("desktop-bird"), None);
    layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
    layer.set_exclusive_zone(-1); // don't reserve space / push other windows
    layer.set_keyboard_interactivity(KeyboardInteractivity::None); // never steal focus

    // An empty input region makes every pointer event pass through to the
    // windows underneath — the bird is purely decorative for now.
    let region = Region::new(&compositor).expect("failed to create wl_region");
    layer.wl_surface().set_input_region(Some(region.wl_region()));

    // Initial commit with no buffer; the compositor replies with a configure.
    layer.commit();

    // SlotPool grows on demand; seed it with a common full-HD-sized buffer.
    let pool = SlotPool::new(1920 * 1080 * 4, &shm).expect("failed to create the shm pool");

    let sprite = Sprite::load_from_env().unwrap_or_else(Sprite::placeholder);

    let mut state = AppState::new(
        RegistryState::new(&globals),
        OutputState::new(&globals, &qh),
        shm,
        pool,
        layer,
        region,
        sprite,
    );

    loop {
        event_queue.blocking_dispatch(&mut state).expect("Wayland dispatch failed");
        if state.exit {
            break;
        }
    }
}
