//! desktop-bird — shared Wayland layer-shell renderer.
//!
//! Puts a transparent, click-through bird on the desktop and flies it around
//! with a placeholder wander. No window tracking yet — that (perching on the
//! active window) is the next slice and is per-compositor (see RESEARCH.md).
//!
//! Works on any compositor that implements wlr-layer-shell — notably KWin
//! (Plasma 6) and cosmic-comp (COSMIC).

mod app;
mod art;
mod brain;
mod bubble;
mod control;
mod cosmic;
mod font;
mod kwin;
mod render;
mod sprite;
mod tracker;

use app::AppState;
use cosmic::CosmicTracker;
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
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `--style-sheet [out.png]` renders every built-in bird style across key
    // poses into one contact-sheet PNG, then exits (no Wayland needed) — for
    // picking a BIRD_STYLE.
    if args.first().map(String::as_str) == Some("--style-sheet") {
        let out = args.get(1).map_or("style-sheet.png", String::as_str);
        match art::write_style_sheet(std::path::Path::new(out)) {
            Ok(()) => println!("desktop-bird: wrote style sheet to {out}"),
            Err(err) => {
                eprintln!("desktop-bird: style sheet failed: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // `--export-sprites [dir]` dumps the procedural bird's frames (in the style
    // selected by BIRD_STYLE) to PNGs in the per-state BIRD_SPRITE_DIR layout,
    // then exits (no Wayland needed). Edit those PNGs and run with
    // BIRD_SPRITE_DIR pointing at them to swap the art.
    if args.first().map(String::as_str) == Some("--export-sprites") {
        let dir = args.get(1).map_or("assets/sprites/placeholder", String::as_str);
        match Sprite::placeholder().write_png_frames(std::path::Path::new(dir)) {
            Ok(()) => println!("desktop-bird: wrote placeholder sprite frames to {dir}"),
            Err(err) => {
                eprintln!("desktop-bird: sprite export failed: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // `--preview <ucsur-text> [out.png]` renders the talking bird with a speech
    // bubble to a PNG and exits (no Wayland needed) — a quick way to eyeball the
    // sitelen-pona rendering. The text must already be UCSUR (the Python side
    // owns Latin→UCSUR); e.g. `python desktop-bird/demo.py | head -1` gives one.
    if args.first().map(String::as_str) == Some("--preview") {
        let text = args.get(1).map_or("", String::as_str);
        let out = args.get(2).map_or("preview.png", String::as_str);
        match write_preview(text, std::path::Path::new(out)) {
            Ok(()) => println!("desktop-bird: wrote bubble preview to {out}"),
            Err(err) => {
                eprintln!("desktop-bird: preview failed: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

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
    //
    // `Overlay` (not `Top`) so the bird stays above all normal windows: several
    // compositors raise focused/newly-mapped windows above the `Top` layer, but
    // the `Overlay` layer sits above the whole normal window stack.
    let surface = compositor.create_surface(&qh);
    let layer =
        layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("desktop-bird"), None);
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

    // Custom PNG art (BIRD_SPRITE_DIR) has no procedural style to retune, so the
    // panel's appearance controls don't apply to it; otherwise track the chosen
    // style so live restyling / amplitude tuning can re-render the bird.
    let (sprite, style) = match Sprite::load_from_env() {
        Some(s) => (s, None),
        None => {
            let st = art::style_from_env();
            (Sprite::procedural(st), Some(st))
        }
    };

    let registry_state = RegistryState::new(&globals);

    // Window tracking (for perching) is per-compositor. COSMIC advertises
    // cosmic-toplevel-info on the Wayland registry; KWin (Plasma 6) has no client
    // protocol for it, so we use a KWin-script → D-Bus bridge (see kwin.rs).
    // Elsewhere the bird just wanders.
    let has_cosmic = globals
        .contents()
        .with_list(|globals| globals.iter().any(|g| g.interface == "zcosmic_toplevel_info_v1"));
    let (tracker, kwin_tracker) = if has_cosmic {
        (CosmicTracker::bind(&registry_state, &qh), None)
    } else if is_kwin() {
        match kwin::start() {
            Some(kt) => (None, Some(kt)),
            None => {
                eprintln!("desktop-bird: KWin detected but script bridge failed; bird will wander");
                (None, None)
            }
        }
    } else {
        eprintln!("desktop-bird: no window-tracking backend; bird will wander");
        (None, None)
    };

    // Speech bubbles are fed from stdin: one bubble per line, UTF-8 sitelen
    // pona (UCSUR) text, optionally `<text>\t<seconds>`. A reader thread parses
    // lines and forwards them to the renderer over the bubble channel; the main
    // Wayland loop drains it without blocking. This is the seam the Python side
    // pipes into — e.g. `python desktop-bird/demo.py | desktop-bird`, and later
    // the Toki Pona model (Latin out → sitelen.latin_to_ucsur → here).
    let (bubble_tx, bubble_rx) = bubble::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break }; // stdin closed / read error
            let (text, secs) = match line.split_once('\t') {
                Some((t, s)) => (t.trim(), s.trim().parse::<f32>().unwrap_or(4.0)),
                None => (line.trim(), 4.0),
            };
            if text.is_empty() {
                continue;
            }
            if bubble_tx.send((text.to_string(), secs)).is_err() {
                break; // receiver dropped — the app is exiting
            }
        }
    });

    // Live tuning: the bird-panel connects over a Unix socket and streams
    // ControlMsgs; the main loop drains them every frame (see control.rs).
    let control_rx = control::start();

    let mut state = AppState::new(
        registry_state,
        OutputState::new(&globals, &qh),
        shm,
        pool,
        layer,
        region,
        sprite,
        style,
        tracker,
        kwin_tracker,
        bubble_rx,
        control_rx,
    );

    loop {
        event_queue.blocking_dispatch(&mut state).expect("Wayland dispatch failed");
        if state.exit {
            break;
        }
    }
}

/// True on a KDE Plasma session, where we drive perching via the KWin script
/// bridge (`kwin.rs`). `XDG_CURRENT_DESKTOP` is a colon-separated list (e.g.
/// "KDE" or "KDE:plasma"); match "KDE" case-insensitively.
fn is_kwin() -> bool {
    std::env::var("XDG_CURRENT_DESKTOP")
        .map(|d| d.split(':').any(|p| p.eq_ignore_ascii_case("KDE")))
        .unwrap_or(false)
}

/// Render the talking bird with a speech bubble into a PNG (no Wayland needed).
/// `text` is UCSUR sitelen pona; empty falls back to a built-in sample.
fn write_preview(text: &str, out: &std::path::Path) -> Result<(), String> {
    use sprite::AnimId;

    // Default sample: "mi moku e kili" already in UCSUR (the Python side owns
    // Latin→UCSUR; this literal is just so `--preview` works with no argument).
    let text = if text.is_empty() { "\u{F1934} \u{F1936} \u{F1909} \u{F191A}" } else { text };

    const W: u32 = 420;
    const H: u32 = 220;
    let mut canvas = vec![0u8; (W * H * 4) as usize];

    // Talking bird near the lower centre, stepped to its beak-open frame.
    let mut sprite = Sprite::placeholder();
    sprite.set_anim(AnimId::Talk);
    sprite.advance(1.0 / 6.0);
    let frame = sprite.current_frame();
    let (fw, fh) = (frame.w as i32, frame.h as i32);
    let bird_x = W as i32 / 2 - fw / 2;
    let bird_y = H as i32 - fh - 20;
    render::blit(&mut canvas, W, H, frame, bird_x, bird_y, false);

    // Speech bubble above it.
    let _ = render::draw_bubble(
        &mut canvas, W, H, text, bird_x, bird_y, fw, fh,
        &bird_protocol::BubbleTuning::default(),
    );

    // Convert premultiplied BGRA → straight RGBA for a PNG with alpha.
    let mut rgba = vec![0u8; canvas.len()];
    for i in (0..canvas.len()).step_by(4) {
        let (b, g, r, a) = (canvas[i], canvas[i + 1], canvas[i + 2], canvas[i + 3]);
        let unpremul = |c: u8| {
            if a == 0 { 0 } else { ((c as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8 }
        };
        rgba[i] = unpremul(r);
        rgba[i + 1] = unpremul(g);
        rgba[i + 2] = unpremul(b);
        rgba[i + 3] = a;
    }
    let img = image::RgbaImage::from_raw(W, H, rgba).ok_or("bad canvas buffer")?;
    img.save(out).map_err(|e| format!("save {}: {e}", out.display()))
}
