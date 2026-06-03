//! COSMIC window tracker.
//!
//! cosmic-comp natively reports per-output toplevel geometry via
//! `cosmic-toplevel-info-unstable-v1`. We use cctk's `ToplevelInfoState`, which
//! is dispatched on the same Wayland event queue as the renderer (the protocol
//! globals live on the same registry), so no extra event loop is needed.
//!
//! NOTE: cctk / cosmic-protocols are GPL-3.0-only, so a build that includes this
//! module is GPL-3.0 as a whole.

use cosmic_client_toolkit::toplevel_info::{ToplevelInfo, ToplevelInfoState};
use cosmic_protocols::toplevel_info::v1::client::zcosmic_toplevel_handle_v1::State;

use crate::render::Rect;
use crate::tracker::{WindowInfo, WindowTracker};

pub struct CosmicTracker {
    /// cctk state; also the handler target for toplevel events (see app.rs).
    pub info_state: ToplevelInfoState,
    /// Cached active window, recomputed whenever toplevel info changes.
    active: Option<WindowInfo>,
}

impl CosmicTracker {
    /// Wrap an already-bound `ToplevelInfoState`. Construction (which needs the
    /// concrete `QueueHandle` type) happens in `main`; see [`ToplevelInfoState::try_new`].
    pub fn new(info_state: ToplevelInfoState) -> Self {
        CosmicTracker { info_state, active: None }
    }

    /// Recompute the cached active window from the current toplevel list. Called
    /// after each new/update/closed toplevel event.
    pub fn refresh(&mut self) {
        let next = self
            .info_state
            .toplevels()
            .find(|t| t.state.contains(&State::Activated))
            .and_then(window_info);

        // Log when the focused window changes (handy for verifying the tracker).
        let changed = match (&self.active, &next) {
            (Some(a), Some(b)) => a.app_id != b.app_id || a.title != b.title,
            (None, None) => false,
            _ => true,
        };
        if changed {
            match &next {
                Some(w) => eprintln!(
                    "desktop-bird: active window -> [{}] \"{}\" at {:?}",
                    w.app_id, w.title, w.geometry
                ),
                None => eprintln!("desktop-bird: active window -> (none)"),
            }
        }

        self.active = next;
    }
}

impl WindowTracker for CosmicTracker {
    fn active_window(&self) -> Option<WindowInfo> {
        self.active.clone()
    }
}

/// Convert a cctk `ToplevelInfo` into our `WindowInfo`, using the geometry on
/// the first output it reports (single-output for now; multi-output later picks
/// the output the bird's surface is on).
fn window_info(t: &ToplevelInfo) -> Option<WindowInfo> {
    let geom = t.geometry.values().next()?;
    Some(WindowInfo {
        geometry: Rect { x: geom.x, y: geom.y, w: geom.width, h: geom.height },
        title: t.title.clone(),
        app_id: t.app_id.clone(),
    })
}
