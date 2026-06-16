//! Application state and Wayland event handling for the shared renderer.
//!
//! One output-sized, transparent, click-through layer surface. The bird is
//! painted at an offset inside that buffer; movement is just redrawing at a new
//! offset, with damage limited to the bird's old+new bounding boxes so the
//! compositor only re-uploads a small region each frame.

use std::num::NonZeroU32;
use std::time::Instant;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{LayerShellHandler, LayerSurface, LayerSurfaceConfigure},
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use wayland_client::{
    protocol::{wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

use std::sync::mpsc::Receiver;

use bird_protocol::{ControlMsg, Tuning};

use crate::art::BirdStyle;
use crate::brain::BirdBrain;
use crate::bubble::BubbleState;
use crate::cosmic::CosmicTracker;
use crate::kwin::KwinTracker;
use crate::render::{self, Rect};
use crate::sprite::{AnimId, Sprite};
use crate::tracker::WindowTracker;

pub struct AppState {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub shm: Shm,
    pub pool: SlotPool,
    pub layer: LayerSurface,
    // Kept alive for as long as the surface uses it. Empty => click-through.
    _input_region: Region,
    pub exit: bool,

    width: u32,
    height: u32,
    configured: bool,

    sprite: Sprite,
    brain: BirdBrain,
    last_frame: Instant,
    /// Dirty rect from the previous frame (bird + any bubble), for damage union.
    prev_rect: Option<Rect>,
    /// Active-window source. `None` when not on COSMIC (the bird just wanders;
    /// the KWin backend lands in a later slice). Updated by the `Dispatch` impls
    /// in `cosmic.rs`, hence `pub(crate)`.
    pub(crate) tracker: Option<CosmicTracker>,
    /// KWin (Plasma 6) active-window source. `Some` when the KWin script bridge
    /// is up; mutually exclusive with `tracker` in practice. Owns the D-Bus
    /// connection + KWin-script lifecycle.
    kwin_tracker: Option<KwinTracker>,
    /// Currently displayed speech bubble, if any.
    bubble: Option<BubbleState>,
    /// Incoming `(text, duration_secs)` messages from the model thread.
    bubble_rx: Receiver<(String, f32)>,
    /// Live tuning from the control panel (defaults reproduce the originals).
    tuning: Tuning,
    /// Current procedural style, or `None` when custom PNG art is loaded — its
    /// poses/colours can't be re-rendered, so the appearance controls no-op
    /// (frame-rate controls still apply).
    style: Option<&'static BirdStyle>,
    /// Incoming control-panel commands.
    control_rx: Receiver<ControlMsg>,
    /// `BIRD_DEBUG`: frame counter + last (state, pose) for throttled tracing.
    dbg_frame: u64,
    dbg_prev: Option<(&'static str, AnimId)>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry_state: RegistryState,
        output_state: OutputState,
        shm: Shm,
        pool: SlotPool,
        layer: LayerSurface,
        input_region: Region,
        sprite: Sprite,
        style: Option<&'static BirdStyle>,
        tracker: Option<CosmicTracker>,
        kwin_tracker: Option<KwinTracker>,
        bubble_rx: Receiver<(String, f32)>,
        control_rx: Receiver<ControlMsg>,
    ) -> Self {
        AppState {
            registry_state,
            output_state,
            shm,
            pool,
            layer,
            _input_region: input_region,
            exit: false,
            width: 0,
            height: 0,
            configured: false,
            sprite,
            brain: BirdBrain::new(),
            last_frame: Instant::now(),
            prev_rect: None,
            tracker,
            kwin_tracker,
            bubble: None,
            bubble_rx,
            tuning: Tuning::default(),
            style,
            control_rx,
            dbg_frame: 0,
            dbg_prev: None,
        }
    }

    /// Apply one live-tuning command from the control panel.
    fn apply_control(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::SetTuning(t) => {
                let anim_changed = t.anim != self.tuning.anim;
                self.brain.set_tuning(t.motion);
                self.tuning = t;
                if anim_changed {
                    self.apply_anim();
                }
            }
            ControlMsg::SetStyle(name) => {
                if let Some(st) = crate::art::style_by_name(&name) {
                    self.style = Some(st);
                    self.rebuild_sprite();
                }
            }
            ControlMsg::Bubble { text, secs } => {
                if !text.trim().is_empty() {
                    self.bubble = Some(BubbleState { text, remaining: secs });
                }
            }
            ControlMsg::Force(s) => self.brain.force(s),
        }
    }

    /// Push the current animation tuning into the sprite. Frame rates apply to
    /// any sprite; pose amplitudes need re-rendering, so they only affect the
    /// procedural bird.
    fn apply_anim(&mut self) {
        let a = self.tuning.anim;
        self.sprite.set_fps(AnimId::Idle, a.fps_idle);
        self.sprite.set_fps(AnimId::Fly, a.fps_fly);
        self.sprite.set_fps(AnimId::Perch, a.fps_perch);
        self.sprite.set_fps(AnimId::Talk, a.fps_talk);
        if self.style.is_some() {
            self.rebuild_sprite();
        }
    }

    /// Re-render the procedural sprite for the current style + animation tuning,
    /// preserving the playing clip. Procedural frames are always 56x40, so the
    /// bird's bounds normally don't change; only re-derive them (which recentres
    /// and restarts wandering) if the canvas size actually did — e.g. switching
    /// from custom PNG art to a built-in style.
    fn rebuild_sprite(&mut self) {
        let Some(st) = self.style else { return };
        let current = self.sprite.current_anim();
        let (ow, oh) = self.sprite.frame_size();
        self.sprite = Sprite::procedural_tuned(st, &self.tuning.anim);
        self.sprite.set_anim(current);
        let (nw, nh) = self.sprite.frame_size();
        if (nw, nh) != (ow, oh) && self.width > 0 && self.height > 0 {
            self.brain.set_bounds(self.width, self.height, nw, nh);
            self.prev_rect = None;
        }
    }

    /// The size (in surface-local pixels) of the output the bird lives on, used
    /// as a fallback when a `configure` reports a 0 dimension. Single-output for
    /// now; takes the first known output.
    fn output_size(&self) -> Option<(u32, u32)> {
        self.output_state.outputs().find_map(|o| {
            let info = self.output_state.info(&o)?;
            if let Some((w, h)) = info.logical_size {
                if w > 0 && h > 0 {
                    return Some((w as u32, h as u32));
                }
            }
            let mode = info.modes.iter().find(|m| m.current)?;
            let scale = info.scale_factor.max(1);
            Some((
                (mode.dimensions.0 / scale).max(1) as u32,
                (mode.dimensions.1 / scale).max(1) as u32,
            ))
        })
    }

    /// Advance simulation + animation by elapsed wall-clock time, then redraw.
    fn tick(&mut self, qh: &QueueHandle<Self>) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25); // clamp after stalls
        self.last_frame = now;

        // Drain any KWin geometry updates first (&mut borrow), then read the
        // active window from whichever backend is live.
        if let Some(kt) = self.kwin_tracker.as_mut() {
            kt.drain();
        }
        let active = self
            .tracker
            .as_ref()
            .and_then(|t| t.active_window())
            .or_else(|| self.kwin_tracker.as_ref().and_then(|t| t.active_window()));
        self.brain.update(dt, active.as_ref());

        // Apply any live-tuning commands from the control panel before drawing.
        while let Ok(msg) = self.control_rx.try_recv() {
            self.apply_control(msg);
        }

        // Accept the most-recent queued bubble message (drain the channel,
        // keep last), then tick the active bubble.
        while let Ok((text, duration)) = self.bubble_rx.try_recv() {
            self.bubble = Some(BubbleState { text, remaining: duration });
        }
        if let Some(b) = &mut self.bubble {
            if !b.tick(dt) {
                self.bubble = None;
            }
        }

        // Pick the body animation. Talk (a perched beak-chirp) must not suppress
        // flight, or a talking bird that's relocating looks like it's floating.
        let pose = choose_pose(self.bubble.is_some(), self.brain.pose());
        self.sprite.set_anim(pose);
        self.sprite.advance(dt);

        // BIRD_DEBUG: trace state/pose/position to stderr — on every state/pose
        // transition, plus a heartbeat every ~20 frames so a silent "float" still
        // shows whether the position is changing and the fly frame is cycling.
        if std::env::var_os("BIRD_DEBUG").is_some() {
            self.dbg_frame = self.dbg_frame.wrapping_add(1);
            let st = self.brain.state_name();
            let changed = self.dbg_prev.is_none_or(|(ps, pp)| ps != st || pp != pose);
            if changed || self.dbg_frame % 20 == 0 {
                let (x, y) = self.brain.position();
                let f = self.dbg_frame;
                let fly = self.sprite.frame_idx();
                eprintln!("bird f{f} state={st} pose={pose:?} pos=({x},{y}) flyframe={fly}");
            }
            self.dbg_prev = Some((st, pose));
        }

        self.draw(qh);
    }

    fn draw(&mut self, qh: &QueueHandle<Self>) {
        if !self.configured || self.width == 0 || self.height == 0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let stride = w as i32 * 4;

        let (buffer, canvas) = self
            .pool
            .create_buffer(w as i32, h as i32, stride, wl_shm::Format::Argb8888)
            .expect("create shm buffer");

        // The surface is transparent everywhere except the bird, so a freshly
        // cleared buffer is always fully correct — undamaged regions stay
        // transparent regardless of how the compositor treats them.
        canvas.fill(0);

        let frame = self.sprite.current_frame();
        let (px, py) = self.brain.position();
        let flip = self.brain.facing_left();
        render::blit(canvas, w, h, frame, px, py, flip);

        let bird_rect = Rect { x: px, y: py, w: frame.w as i32, h: frame.h as i32 };

        // Draw the speech bubble (if any) and extend the dirty region.
        let new_rect = if let Some(bubble) = &self.bubble {
            let bubble_rect = render::draw_bubble(
                canvas, w, h,
                &bubble.text,
                px, py, frame.w as i32, frame.h as i32,
                &self.tuning.bubble,
            );
            bird_rect.union(&bubble_rect).clamp(w, h)
        } else {
            bird_rect.clamp(w, h)
        };

        let damage = match self.prev_rect {
            Some(prev) => prev.union(&new_rect).clamp(w, h),
            None => new_rect,
        };
        self.prev_rect = Some(new_rect);

        // Clone the surface handle to avoid borrowing `self.layer` across the
        // attach/commit calls below.
        let surface = self.layer.wl_surface().clone();
        if damage.w > 0 && damage.h > 0 {
            surface.damage_buffer(damage.x, damage.y, damage.w, damage.h);
        }
        // Schedule the next frame; the compositor throttles us to its refresh.
        surface.frame(qh, surface.clone());
        buffer.attach_to(&surface).expect("attach buffer");
        surface.commit();
    }
}

/// Choose the body animation given whether a speech bubble is showing and the
/// brain's movement pose. `Talk` is a grounded beak-chirp, so while the bird is
/// flying we keep flapping (the bubble still floats along above it) and only
/// chirp when it's perched or paused.
fn choose_pose(bubble: bool, body: AnimId) -> AnimId {
    match (bubble, body) {
        (true, AnimId::Fly) => AnimId::Fly, // flapping wins; bubble still shows
        (true, _) => AnimId::Talk,          // perched / idle → chirp
        (false, body) => body,
    }
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        self.tick(qh);
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for AppState {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        // wlr-layer-shell lets the compositor send 0 for a dimension, meaning
        // "you choose" — KWin/cosmic-comp send the full output size for an
        // all-edges-anchored surface, but if either axis ever comes back 0 we
        // must NOT keep the initial 0 (which leaves the surface unconfigured and
        // the bird invisible). Fall back to the output's own size.
        let (ow, oh) = self.output_size().unwrap_or((self.width, self.height));
        let new_w = NonZeroU32::new(configure.new_size.0).map_or(ow, NonZeroU32::get);
        let new_h = NonZeroU32::new(configure.new_size.1).map_or(oh, NonZeroU32::get);

        let size_changed = new_w != self.width || new_h != self.height;
        self.width = new_w;
        self.height = new_h;

        if size_changed && self.width > 0 && self.height > 0 {
            let (sw, sh) = self.sprite.frame_size();
            self.brain.set_bounds(self.width, self.height, sw, sh);
            self.prev_rect = None; // dimensions changed; force a clean redraw
        }

        if !self.configured && self.width > 0 && self.height > 0 {
            self.configured = true;
            self.last_frame = Instant::now();
            self.draw(qh);
        }
    }
}

impl OutputHandler for AppState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ShmHandler for AppState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for AppState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_compositor!(AppState);
delegate_output!(AppState);
delegate_shm!(AppState);
delegate_layer!(AppState);
delegate_registry!(AppState);
// The COSMIC toplevel-info `Dispatch` impls live in `cosmic.rs`.

#[cfg(test)]
mod tests {
    use super::choose_pose;
    use crate::sprite::AnimId;

    #[test]
    fn talking_while_flying_keeps_flapping() {
        // The bug: a bubble used to force Talk even mid-flight, so a talking bird
        // relocating to a window looked like it was floating. Flying must win.
        assert_eq!(choose_pose(true, AnimId::Fly), AnimId::Fly);
    }

    #[test]
    fn talking_while_grounded_chirps() {
        assert_eq!(choose_pose(true, AnimId::Perch), AnimId::Talk);
        assert_eq!(choose_pose(true, AnimId::Idle), AnimId::Talk);
    }

    #[test]
    fn silent_bird_uses_its_movement_pose() {
        for body in [AnimId::Fly, AnimId::Perch, AnimId::Idle] {
            assert_eq!(choose_pose(false, body), body);
        }
    }
}
