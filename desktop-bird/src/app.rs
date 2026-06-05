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

use crate::brain::BirdBrain;
use crate::bubble::BubbleState;
use crate::cosmic::CosmicTracker;
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
    /// Currently displayed speech bubble, if any.
    bubble: Option<BubbleState>,
    /// Incoming `(text, duration_secs)` messages from the model thread.
    bubble_rx: Receiver<(String, f32)>,
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
        tracker: Option<CosmicTracker>,
        bubble_rx: Receiver<(String, f32)>,
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
            bubble: None,
            bubble_rx,
        }
    }

    /// Advance simulation + animation by elapsed wall-clock time, then redraw.
    fn tick(&mut self, qh: &QueueHandle<Self>) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25); // clamp after stalls
        self.last_frame = now;

        let active = self.tracker.as_ref().and_then(|t| t.active_window());
        self.brain.update(dt, active.as_ref());

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

        // Chirp (Talk) while a bubble is up; otherwise follow the brain's pose.
        let pose = if self.bubble.is_some() { AnimId::Talk } else { self.brain.pose() };
        self.sprite.set_anim(pose);
        self.sprite.advance(dt);

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
        let new_w = NonZeroU32::new(configure.new_size.0).map_or(self.width, NonZeroU32::get);
        let new_h = NonZeroU32::new(configure.new_size.1).map_or(self.height, NonZeroU32::get);

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
