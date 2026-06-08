//! KWin (Plasma 6) window tracker (MIT).
//!
//! KWin exposes no client protocol for window geometry, so the supported route
//! is a KWin script (`kwin/perch.js`) that runs inside the compositor and pushes
//! the active window's `frameGeometry` out via `callDBus` to a session-bus
//! service we register here. We self-load the script through
//! `org.kde.kwin.Scripting`, so the user just runs the bird — no manual install.
//!
//! Unlike the COSMIC backend (driven by Wayland events on the renderer's own
//! queue), updates arrive on a background D-Bus thread and reach the renderer
//! over an mpsc channel that `AppState::tick()` drains each frame. That relies on
//! the continuous frame-callback heartbeat (the bird always re-arms a frame), so
//! no separate event loop is needed; if frame callbacks ever stop (full
//! occlusion / DPMS) the bird freezes at its last geometry — but it isn't visible
//! then anyway.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

use zbus::blocking::{connection, Connection};

use crate::render::Rect;
use crate::tracker::{WindowInfo, WindowTracker};

const SERVICE: &str = "org.waso.DesktopBird";
const PATH: &str = "/Tracker";
const SCRIPT_NAME: &str = "desktop-bird";
const SCRIPT_SRC: &str = include_str!("../kwin/perch.js");

// org.kde.kwin.Scripting, for self-loading the script.
const KWIN: &str = "org.kde.KWin";
const SCRIPTING_PATH: &str = "/Scripting";
const SCRIPTING_IFACE: &str = "org.kde.kwin.Scripting";

/// The D-Bus interface the KWin script calls into. Each call forwards an
/// `Option<WindowInfo>` to the renderer over the channel. `Sender` isn't reliably
/// `Sync`, and the object server needs the interface `Send + Sync`, so the sender
/// lives behind a `Mutex` (uncontended — zbus serialises calls on its executor).
struct TrackerIface {
    tx: Mutex<Sender<Option<WindowInfo>>>,
}

#[zbus::interface(name = "org.waso.DesktopBird.Tracker")]
impl TrackerIface {
    #[zbus(name = "UpdateWindow")]
    fn update_window(&self, x: i32, y: i32, w: i32, h: i32, title: String, app_id: String) {
        let info = WindowInfo { geometry: Rect { x, y, w, h }, title, app_id };
        let _ = self.tx.lock().unwrap().send(Some(info));
    }

    #[zbus(name = "ClearWindow")]
    fn clear_window(&self) {
        let _ = self.tx.lock().unwrap().send(None);
    }
}

/// Active-window source for KWin. Owns the bus connection (the served interface
/// stays alive only while the connection does) and the receiver the renderer
/// drains each frame.
pub struct KwinTracker {
    conn: Connection,
    rx: Receiver<Option<WindowInfo>>,
    active: Option<WindowInfo>,
}

/// Register the service, write + self-load the KWin script. Returns `None` if any
/// step fails (caller falls back to wandering).
pub fn start() -> Option<KwinTracker> {
    let (tx, rx) = mpsc::channel();

    // Own the name + serve the interface BEFORE loading the script, so the
    // script's first `callDBus` has a live target. `build()` acquires the name
    // synchronously; zbus's internal executor thread then dispatches calls.
    let conn = connection::Builder::session()
        .ok()?
        .name(SERVICE)
        .ok()?
        .serve_at(PATH, TrackerIface { tx: Mutex::new(tx) })
        .ok()?
        .build()
        .ok()?;

    let path = write_script()?;
    if load_script(&conn, &path).is_none() {
        eprintln!("desktop-bird: failed to load KWin script via org.kde.kwin.Scripting");
        return None;
    }

    eprintln!("desktop-bird: KWin perch tracker active (script loaded)");
    Some(KwinTracker { conn, rx, active: None })
}

fn write_script() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("desktop-bird");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("perch.js");
    std::fs::write(&path, SCRIPT_SRC).ok()?;
    Some(path)
}

/// Call a method on `org.kde.kwin.Scripting`. Inlined (not generic) so we don't
/// have to name zbus's `serde::Serialize + DynamicType` body bound.
macro_rules! scripting_call {
    ($conn:expr, $method:expr, $body:expr) => {
        $conn.call_method(Some(KWIN), SCRIPTING_PATH, Some(SCRIPTING_IFACE), $method, $body)
    };
}

fn load_script(conn: &Connection, path: &Path) -> Option<()> {
    let p = path.to_str()?;
    // Unload-first (best effort) so reloads don't stack duplicate script
    // instances; keyed by the plugin name we pass to the `ss` loadScript overload.
    let _ = scripting_call!(conn, "unloadScript", &(SCRIPT_NAME,));
    let id: i32 = scripting_call!(conn, "loadScript", &(p, SCRIPT_NAME))
        .ok()?
        .body()
        .deserialize()
        .ok()?;
    if id < 0 {
        return None;
    }
    scripting_call!(conn, "start", &()).ok()?;
    Some(())
}

impl KwinTracker {
    /// Drain queued geometry updates, keeping the most recent, and log focus
    /// changes (matching the COSMIC backend's `refresh`). Called once per frame.
    pub fn drain(&mut self) {
        while let Ok(next) = self.rx.try_recv() {
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
}

impl WindowTracker for KwinTracker {
    fn active_window(&self) -> Option<WindowInfo> {
        self.active.clone()
    }
}

impl Drop for KwinTracker {
    fn drop(&mut self) {
        // Best-effort: unload the script so it doesn't linger in KWin calling a
        // dead service. (A hard kill/SIGINT skips this; the next launch's
        // unload-first self-heals, and the stray callDBus is harmless.)
        let _ = scripting_call!(self.conn, "unloadScript", &(SCRIPT_NAME,));
    }
}
