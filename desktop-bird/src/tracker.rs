//! Abstraction over "where is the active window" — the one capability that is
//! not portable across Wayland compositors (see RESEARCH.md §3).
//!
//! Each desktop environment supplies geometry through a different channel, so
//! the renderer/brain depend only on this trait. Backends:
//!   - [`crate::cosmic::CosmicTracker`] — COSMIC, via `cosmic-toplevel-info`.
//!   - KWin (`.kwinscript` → D-Bus) — future slice.

use crate::render::Rect;

/// The active window's geometry and identity, in output-local logical pixels
/// (same coordinate space as the bird's per-output buffer).
#[derive(Clone, Debug)]
pub struct WindowInfo {
    pub geometry: Rect,
    pub title: String,
    pub app_id: String,
}

/// A source of the currently active (focused) window for the bird to perch on.
pub trait WindowTracker {
    /// The active window, or `None` if nothing is focused / no geometry is known.
    fn active_window(&self) -> Option<WindowInfo>;
}
