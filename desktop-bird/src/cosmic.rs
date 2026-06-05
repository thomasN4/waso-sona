//! COSMIC window tracker (MIT).
//!
//! cosmic-comp reports per-output toplevel geometry via
//! `cosmic-toplevel-info-unstable-v1`. To stay MIT (cctk / cosmic-protocols are
//! GPL-3.0), we generate our own client bindings from the vendored HPND XML in
//! `protocols/` with `wayland-scanner`, and hand-write the small slice of
//! `Dispatch` we need: enumerate toplevels via the standard
//! `ext-foreign-toplevel-list`, request a cosmic handle per toplevel, and read
//! its `state` + `geometry`.
//!
//! Everything is dispatched on the renderer's own Wayland event queue (the
//! globals live on the same registry), so no extra event loop is needed yet.

use smithay_client_toolkit::registry::RegistryState;
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::ext::foreign_toplevel_list::v1::client::{
    ext_foreign_toplevel_handle_v1::{self, ExtForeignToplevelHandleV1},
    ext_foreign_toplevel_list_v1::{self, ExtForeignToplevelListV1},
};

use crate::app::AppState;
use crate::render::Rect;
use crate::tracker::{WindowInfo, WindowTracker};

/// Generated client bindings for the vendored cosmic-toplevel-info XML.
#[allow(
    non_camel_case_types,
    non_upper_case_globals,
    unused_imports,
    dead_code,
    clippy::all
)]
pub mod proto {
    // Shim around the real crate: wayland-scanner emits `super::wayland_client::
    // ObjectId` for the untyped workspace object args (see the XML note), but the
    // real crate only re-exports `ObjectId` publicly under `backend`. Re-export
    // it at this shim's root so that path resolves. Everything else passes
    // through unchanged.
    mod wayland_client {
        pub use ::wayland_client::backend::ObjectId;
        pub use ::wayland_client::*;
    }
    use wayland_client::protocol::*;
    use wayland_protocols::ext::foreign_toplevel_list::v1::client::*;

    pub mod __interfaces {
        use wayland_client::protocol::__interfaces::*;
        use wayland_protocols::ext::foreign_toplevel_list::v1::client::__interfaces::*;
        wayland_scanner::generate_interfaces!("protocols/cosmic-toplevel-info-unstable-v1.xml");
    }
    use self::__interfaces::*;

    wayland_scanner::generate_client_code!("protocols/cosmic-toplevel-info-unstable-v1.xml");
}

use proto::zcosmic_toplevel_handle_v1::{self, ZcosmicToplevelHandleV1};
use proto::zcosmic_toplevel_info_v1::{self, ZcosmicToplevelInfoV1};

/// `activated` value from the toplevel `state` enum (see the XML).
const STATE_ACTIVATED: u32 = 2;

/// Marker user-data for the protocol objects we create.
#[derive(Debug)]
pub struct GlobalData;

/// One tracked toplevel, assembled from events across both handles.
struct Toplevel {
    foreign: ExtForeignToplevelHandleV1,
    cosmic: Option<ZcosmicToplevelHandleV1>,
    title: String,
    app_id: String,
    activated: bool,
    geometry: Option<Rect>,
}

pub struct CosmicTracker {
    info: ZcosmicToplevelInfoV1,
    _list: ExtForeignToplevelListV1,
    toplevels: Vec<Toplevel>,
    /// Cached active window, recomputed whenever toplevel info changes.
    active: Option<WindowInfo>,
}

impl CosmicTracker {
    /// Bind the foreign-toplevel-list and cosmic-toplevel-info globals. Returns
    /// `None` if either is missing (i.e. not COSMIC, or too old for geometry).
    pub fn bind<D>(registry: &RegistryState, qh: &QueueHandle<D>) -> Option<Self>
    where
        D: Dispatch<ExtForeignToplevelListV1, GlobalData>
            + Dispatch<ZcosmicToplevelInfoV1, GlobalData>
            + 'static,
    {
        let list =
            registry.bind_one::<ExtForeignToplevelListV1, _, _>(qh, 1..=1, GlobalData).ok()?;
        // v2+ carries the `geometry` event we rely on.
        let info =
            registry.bind_one::<ZcosmicToplevelInfoV1, _, _>(qh, 2..=3, GlobalData).ok()?;
        Some(CosmicTracker { info, _list: list, toplevels: Vec::new(), active: None })
    }

    fn entry_by_foreign(&mut self, h: &ExtForeignToplevelHandleV1) -> Option<&mut Toplevel> {
        self.toplevels.iter_mut().find(|t| &t.foreign == h)
    }

    fn entry_by_cosmic(&mut self, h: &ZcosmicToplevelHandleV1) -> Option<&mut Toplevel> {
        self.toplevels.iter_mut().find(|t| t.cosmic.as_ref() == Some(h))
    }

    /// Recompute the cached active window and log focus changes.
    fn refresh(&mut self) {
        let next = self.toplevels.iter().find(|t| t.activated).and_then(|t| {
            t.geometry.map(|geometry| WindowInfo {
                geometry,
                title: t.title.clone(),
                app_id: t.app_id.clone(),
            })
        });

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

// --- Dispatch: ext-foreign-toplevel-list (enumerate toplevels) ------------

impl Dispatch<ExtForeignToplevelListV1, GlobalData> for AppState {
    fn event(
        state: &mut Self,
        _list: &ExtForeignToplevelListV1,
        event: ext_foreign_toplevel_list_v1::Event,
        _data: &GlobalData,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let Some(tracker) = state.tracker.as_mut() else { return };
        if let ext_foreign_toplevel_list_v1::Event::Toplevel { toplevel } = event {
            // Request the cosmic extension handle so we get geometry + state.
            let cosmic = tracker.info.get_cosmic_toplevel(&toplevel, qh, GlobalData);
            tracker.toplevels.push(Toplevel {
                foreign: toplevel,
                cosmic: Some(cosmic),
                title: String::new(),
                app_id: String::new(),
                activated: false,
                geometry: None,
            });
        }
    }

    wayland_client::event_created_child!(AppState, ExtForeignToplevelListV1, [
        ext_foreign_toplevel_list_v1::EVT_TOPLEVEL_OPCODE => (ExtForeignToplevelHandleV1, GlobalData),
    ]);
}

// --- Dispatch: ext foreign toplevel handle (title / app_id / lifetime) -----

impl Dispatch<ExtForeignToplevelHandleV1, GlobalData> for AppState {
    fn event(
        state: &mut Self,
        handle: &ExtForeignToplevelHandleV1,
        event: ext_foreign_toplevel_handle_v1::Event,
        _data: &GlobalData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(tracker) = state.tracker.as_mut() else { return };
        match event {
            ext_foreign_toplevel_handle_v1::Event::Title { title } => {
                if let Some(entry) = tracker.entry_by_foreign(handle) {
                    entry.title = title;
                }
            }
            ext_foreign_toplevel_handle_v1::Event::AppId { app_id } => {
                if let Some(entry) = tracker.entry_by_foreign(handle) {
                    entry.app_id = app_id;
                }
            }
            ext_foreign_toplevel_handle_v1::Event::Closed => {
                if let Some(pos) = tracker.toplevels.iter().position(|t| &t.foreign == handle) {
                    let entry = tracker.toplevels.remove(pos);
                    if let Some(cosmic) = &entry.cosmic {
                        cosmic.destroy();
                    }
                    entry.foreign.destroy();
                }
                tracker.refresh();
            }
            ext_foreign_toplevel_handle_v1::Event::Done => tracker.refresh(),
            _ => {}
        }
    }
}

// --- Dispatch: cosmic toplevel info (batch done) --------------------------

impl Dispatch<ZcosmicToplevelInfoV1, GlobalData> for AppState {
    fn event(
        state: &mut Self,
        _info: &ZcosmicToplevelInfoV1,
        event: zcosmic_toplevel_info_v1::Event,
        _data: &GlobalData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(tracker) = state.tracker.as_mut() else { return };
        if let zcosmic_toplevel_info_v1::Event::Done = event {
            tracker.refresh();
        }
    }

    // The deprecated v1 `toplevel` event creates a child; never fires at v2+,
    // but the specialization is required because the event carries a new_id.
    wayland_client::event_created_child!(AppState, ZcosmicToplevelInfoV1, [
        zcosmic_toplevel_info_v1::EVT_TOPLEVEL_OPCODE => (ZcosmicToplevelHandleV1, GlobalData),
    ]);
}

// --- Dispatch: cosmic toplevel handle (state + geometry) ------------------

impl Dispatch<ZcosmicToplevelHandleV1, GlobalData> for AppState {
    fn event(
        state: &mut Self,
        handle: &ZcosmicToplevelHandleV1,
        event: zcosmic_toplevel_handle_v1::Event,
        _data: &GlobalData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(tracker) = state.tracker.as_mut() else { return };
        match event {
            zcosmic_toplevel_handle_v1::Event::State { state: states } => {
                let activated = states
                    .chunks_exact(4)
                    .any(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) == STATE_ACTIVATED);
                if let Some(entry) = tracker.entry_by_cosmic(handle) {
                    entry.activated = activated;
                }
                tracker.refresh();
            }
            zcosmic_toplevel_handle_v1::Event::Geometry { x, y, width, height, .. } => {
                if let Some(entry) = tracker.entry_by_cosmic(handle) {
                    entry.geometry = Some(Rect { x, y, w: width, h: height });
                }
                tracker.refresh();
            }
            _ => {}
        }
    }
}
