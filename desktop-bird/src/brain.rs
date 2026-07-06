//! BirdBrain — the bird's behaviour state machine.
//!
//! Pure logic: each tick takes `dt` and the active window's geometry (if any)
//! and updates position / facing / which animation to play. No Wayland here, so
//! it is unit-testable headless (see the tests at the bottom).
//!
//! ```text
//! Wander  (no active window): roam to random points, with idle pauses
//!    └─ window appears ───────────────────────────> Approach
//! Approach (fly to the perch on the window's top edge)
//!    └─ arrived ───────────────────────────────────> Perch
//! Perch   (sit; follow small window moves; idle/fidget anim)
//!    ├─ flit timer elapses ────────────────────────> Flit
//!    └─ window jumps far ──────────────────────────> Approach
//! Flit    (short hop near the window)
//!    └─ arrived ───────────────────────────────────> Approach (back to perch)
//! (any) window gone ───────────────────────────────> Wander
//! ```

use rand::RngExt;

use bird_protocol::{ForceState, MotionTuning};

use crate::render::Rect;
use crate::sprite::AnimId;
use crate::tracker::WindowInfo;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Wander,
    Approach,
    Perch,
    Flit,
}

pub struct BirdBrain {
    /// Current top-left position (buffer pixels).
    x: f32,
    y: f32,
    /// Current target top-left.
    tx: f32,
    ty: f32,
    /// Inclusive upper bounds for the top-left, so the bird stays fully visible.
    max_x: f32,
    max_y: f32,
    sprite_w: f32,
    sprite_h: f32,
    state: State,
    /// Wander pause remaining (s).
    idle: f32,
    /// Seconds left before the next flit while perched.
    flit_timer: f32,
    /// Seconds left to keep flapping after following a window move while
    /// perched (see `tuning.follow_flap_linger`).
    flap_linger: f32,
    facing_left: bool,
    /// Bounds set (surface configured) yet?
    ready: bool,
    /// Live-tunable movement/timing knobs (defaults reproduce the originals).
    tuning: MotionTuning,
}

impl BirdBrain {
    pub fn new() -> Self {
        BirdBrain {
            x: 0.0,
            y: 0.0,
            tx: 0.0,
            ty: 0.0,
            max_x: 0.0,
            max_y: 0.0,
            sprite_w: 0.0,
            sprite_h: 0.0,
            state: State::Wander,
            idle: 0.0,
            flit_timer: 0.0,
            flap_linger: 0.0,
            facing_left: false,
            ready: false,
            tuning: MotionTuning::default(),
        }
    }

    /// Replace the live movement/timing tuning (from the control panel).
    pub fn set_tuning(&mut self, t: MotionTuning) {
        self.tuning = t;
    }

    /// Force a behaviour state for testing. Approach/Perch/Flit only stay put
    /// while a window is focused; with none active the next `update` falls back
    /// to wandering. `Wander` always takes effect.
    pub fn force(&mut self, s: ForceState) {
        if !self.ready {
            return;
        }
        match s {
            ForceState::Wander => self.enter_wander(),
            ForceState::Approach => self.state = State::Approach,
            ForceState::Perch => {
                self.state = State::Perch;
                self.flap_linger = 0.0;
                self.flit_timer = self.rand_flit_delay();
            }
            ForceState::Flit => {
                let here = (self.x, self.y);
                self.start_flit(here);
            }
        }
    }

    /// (Re)configure for a `surf_w x surf_h` surface holding a `sprite_w x
    /// sprite_h` bird. Centres the bird and starts wandering.
    pub fn set_bounds(&mut self, surf_w: u32, surf_h: u32, sprite_w: u32, sprite_h: u32) {
        self.max_x = surf_w.saturating_sub(sprite_w).max(1) as f32;
        self.max_y = surf_h.saturating_sub(sprite_h).max(1) as f32;
        self.sprite_w = sprite_w as f32;
        self.sprite_h = sprite_h as f32;
        self.x = (self.max_x / 2.0).min(self.max_x);
        self.y = (self.max_y / 2.0).min(self.max_y);
        self.ready = true;
        self.enter_wander();
    }

    /// Advance the simulation by `dt` seconds given the active window (if any).
    pub fn update(&mut self, dt: f32, active: Option<&WindowInfo>) {
        if !self.ready {
            return;
        }
        match active {
            None => {
                if self.state != State::Wander {
                    self.enter_wander();
                }
                self.update_wander(dt);
            }
            Some(window) => self.update_with_window(dt, self.perch_point(&window.geometry)),
        }
    }

    pub fn position(&self) -> (i32, i32) {
        (self.x.round() as i32, self.y.round() as i32)
    }

    pub fn facing_left(&self) -> bool {
        self.facing_left
    }

    /// Human-readable current state, for `BIRD_DEBUG` tracing.
    pub fn state_name(&self) -> &'static str {
        match self.state {
            State::Wander => "wander",
            State::Approach => "approach",
            State::Perch => "perch",
            State::Flit => "flit",
        }
    }

    /// Which animation clip the renderer should play.
    pub fn pose(&self) -> AnimId {
        match self.state {
            State::Perch if self.flap_linger > 0.0 => AnimId::Fly,
            State::Perch => AnimId::Perch,
            State::Wander if self.idle > 0.0 => AnimId::Idle,
            _ => AnimId::Fly,
        }
    }

    // --- internals --------------------------------------------------------

    /// Perch target for a window: centred on its top edge, feet at the edge.
    fn perch_point(&self, g: &Rect) -> (f32, f32) {
        let x = g.x as f32 + g.w as f32 / 2.0 - self.sprite_w / 2.0;
        let y = g.y as f32 - self.sprite_h;
        (x.clamp(0.0, self.max_x), y.clamp(0.0, self.max_y))
    }

    fn enter_wander(&mut self) {
        self.state = State::Wander;
        self.idle = 0.0;
        self.pick_wander_target();
    }

    fn pick_wander_target(&mut self) {
        let mut rng = rand::rng();
        self.tx = rng.random_range(0.0..=self.max_x);
        self.ty = rng.random_range(0.0..=self.max_y);
    }

    /// A random flit delay in `[min, max)`, tolerant of a collapsed range
    /// (panel sliders can set min == max).
    fn rand_flit_delay(&self) -> f32 {
        let (lo, hi) = (self.tuning.flit_delay_min, self.tuning.flit_delay_max);
        if hi > lo { rand::rng().random_range(lo..hi) } else { lo }
    }

    fn update_wander(&mut self, dt: f32) {
        if self.idle > 0.0 {
            self.idle -= dt;
            return;
        }
        if self.step_toward(dt) {
            let mut rng = rand::rng();
            if rng.random_bool(self.tuning.wander_idle_chance.clamp(0.0, 1.0)) {
                let (lo, hi) = (self.tuning.wander_idle_min, self.tuning.wander_idle_max);
                self.idle = if hi > lo { rng.random_range(lo..hi) } else { lo };
            }
            self.pick_wander_target();
        }
    }

    fn update_with_window(&mut self, dt: f32, perch: (f32, f32)) {
        // Decide target / transitions relative to the (possibly moved) window.
        match self.state {
            State::Wander => {
                self.state = State::Approach;
                self.tx = perch.0;
                self.ty = perch.1;
            }
            State::Approach => {
                self.tx = perch.0;
                self.ty = perch.1;
            }
            State::Perch => {
                let moved = (perch.0 - self.tx).hypot(perch.1 - self.ty);
                if moved > self.tuning.follow_eps {
                    self.state = State::Approach;
                }
                self.tx = perch.0;
                self.ty = perch.1;
            }
            State::Flit => {} // keep heading to the flit point; ignore window moves
        }

        // Advance the active state.
        match self.state {
            State::Approach => {
                if self.step_toward(dt) {
                    self.state = State::Perch;
                    self.flit_timer = self.rand_flit_delay();
                }
            }
            State::Perch => {
                let (px, py) = (self.x, self.y);
                self.step_toward(dt); // ease along if the window is drifting
                if (self.x - px).abs() > 0.01 || (self.y - py).abs() > 0.01 {
                    self.flap_linger = self.tuning.follow_flap_linger;
                } else {
                    self.flap_linger = (self.flap_linger - dt).max(0.0);
                }
                self.flit_timer -= dt;
                if self.flit_timer <= 0.0 {
                    self.start_flit(perch);
                }
            }
            State::Flit => {
                if self.step_toward(dt) {
                    // Head back to the perch.
                    self.state = State::Approach;
                    self.tx = perch.0;
                    self.ty = perch.1;
                }
            }
            State::Wander => {}
        }
    }

    fn start_flit(&mut self, perch: (f32, f32)) {
        let mut rng = rand::rng();
        // Stray around the perch, biased upward (birds flit up/around, not down
        // into the window). A zero radius (panel slider) means "hop in place".
        let r = self.tuning.flit_radius;
        let dx = if r > 0.0 { rng.random_range(-r..r) } else { 0.0 };
        let dy = if r > 0.0 { rng.random_range(-r..r / 3.0) } else { 0.0 };
        self.tx = (perch.0 + dx).clamp(0.0, self.max_x);
        self.ty = (perch.1 + dy).clamp(0.0, self.max_y);
        self.state = State::Flit;
    }

    /// Move toward `(tx, ty)` at `tuning.speed`, updating facing. Returns whether the
    /// target was reached this step.
    fn step_toward(&mut self, dt: f32) -> bool {
        let dx = self.tx - self.x;
        let dy = self.ty - self.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dx.abs() > 0.5 {
            self.facing_left = dx < 0.0;
        }
        if dist <= self.tuning.arrive_eps {
            self.x = self.tx;
            self.y = self.ty;
            return true;
        }
        let step = self.tuning.speed * dt;
        if step >= dist {
            self.x = self.tx;
            self.y = self.ty;
            return true;
        }
        self.x += dx / dist * step;
        self.y += dy / dist * step;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(x: i32, y: i32, w: i32, h: i32) -> WindowInfo {
        WindowInfo { geometry: Rect { x, y, w, h }, title: String::new(), app_id: String::new() }
    }

    fn brain() -> BirdBrain {
        let mut b = BirdBrain::new();
        b.set_bounds(1000, 1000, 40, 30);
        b
    }

    fn in_bounds(b: &BirdBrain) -> bool {
        let (x, y) = b.position();
        x >= 0 && y >= 0 && x as f32 <= b.max_x && y as f32 <= b.max_y
    }

    #[test]
    fn no_window_wanders_in_bounds() {
        let mut b = brain();
        for _ in 0..400 {
            b.update(0.05, None);
            assert!(in_bounds(&b));
        }
        assert_eq!(b.state, State::Wander);
    }

    #[test]
    fn flies_to_and_perches_on_window() {
        let mut b = brain();
        let w = win(400, 300, 200, 150);
        // ~3 s: enough to reach the perch, before the >=4 s flit timer.
        for _ in 0..60 {
            b.update(0.05, Some(&w));
        }
        assert_eq!(b.state, State::Perch);
        assert_eq!(b.pose(), AnimId::Perch);
        let (px, py) = b.position();
        // Perch = top-edge centre, feet on edge: (400+100-20, 300-30).
        assert!((px - 480).abs() <= 2, "x={px}");
        assert!((py - 270).abs() <= 2, "y={py}");
    }

    #[test]
    fn re_approaches_when_window_jumps() {
        let mut b = brain();
        let w = win(400, 300, 200, 150);
        for _ in 0..60 {
            b.update(0.05, Some(&w));
        }
        assert_eq!(b.state, State::Perch);

        let moved = win(50, 600, 200, 150); // far jump
        b.update(0.05, Some(&moved));
        assert_eq!(b.state, State::Approach);

        for _ in 0..80 {
            b.update(0.05, Some(&moved));
        }
        assert_eq!(b.state, State::Perch);
        let (px, _) = b.position();
        assert!((px - 130).abs() <= 2, "x={px}"); // 50+100-20
    }

    #[test]
    fn flaps_while_following_a_dragged_window() {
        let mut b = brain();
        let mut w = win(400, 300, 200, 150);
        for _ in 0..60 {
            b.update(0.05, Some(&w));
        }
        assert_eq!(b.state, State::Perch);
        assert_eq!(b.pose(), AnimId::Perch);

        // Drag downward in small steps, as interactive moves arrive.
        for _ in 0..20 {
            w.geometry.y += 4;
            b.update(0.05, Some(&w));
            assert_eq!(b.state, State::Perch, "small moves should not re-approach");
            assert_eq!(b.pose(), AnimId::Fly, "should flap while sliding along");
        }

        // Drag stops: settle back to the perch pose once the linger drains.
        for _ in 0..10 {
            b.update(0.05, Some(&w));
        }
        assert_eq!(b.state, State::Perch);
        assert_eq!(b.pose(), AnimId::Perch);
    }

    #[test]
    fn returns_to_wander_when_window_closes() {
        let mut b = brain();
        let w = win(400, 300, 200, 150);
        for _ in 0..60 {
            b.update(0.05, Some(&w));
        }
        assert_eq!(b.state, State::Perch);
        b.update(0.05, None);
        assert_eq!(b.state, State::Wander);
    }

    /// Reproduce the real app loop: `update` is called every frame (~60 fps),
    /// but the compositor reports the window position only sparsely (here every
    /// `update_every` frames). Drive the full pose→sprite pipeline and report,
    /// for the frames where the bird actually moved, what pose it showed and
    /// whether the fly animation cycled. This is the case the per-frame
    /// `flaps_while_following_a_dragged_window` test doesn't cover.
    #[test]
    fn flaps_while_following_sparsely_updated_window() {
        use crate::sprite::{AnimId, Sprite};
        let dt = 1.0 / 60.0;
        let mut b = BirdBrain::new();
        // Disable random flits so the only motion under test is the window-follow.
        // (A flit's brief landing frame — where the state flips to Perch as the
        // bird snaps the last pixel — would otherwise add nondeterministic noise.)
        b.set_tuning(MotionTuning { flit_delay_min: 1.0e6, flit_delay_max: 1.0e6, ..MotionTuning::default() });
        let mut sprite = Sprite::procedural(crate::art::style_from_env());
        let (sw, sh) = sprite.frame_size();
        b.set_bounds(1920, 1080, sw, sh);

        // Settle onto a window first.
        let mut w = win(800, 500, 400, 300);
        for _ in 0..240 {
            b.update(dt, Some(&w));
        }
        assert_eq!(b.state, State::Perch);

        // Now drag the window leftward across the screen; the compositor only
        // reports a new position every `update_every` frames (so most frames
        // re-use the same geometry, exactly like the live app).
        let update_every = 6; // ~10 Hz position updates at 60 fps
        let mut moving_frames = 0;
        let mut moving_but_not_fly = 0;
        let mut fly_indices = std::collections::BTreeSet::new();
        for f in 0..600 {
            if f % update_every == 0 && w.geometry.x > 60 {
                w.geometry.x -= 30; // 30 px per reported step ≈ 300 px/s drag
            }
            let (before_x, before_y) = b.position();
            b.update(dt, Some(&w));
            let (after_x, after_y) = b.position();

            let pose = b.pose();
            sprite.set_anim(pose);
            sprite.advance(dt);

            let moved = (after_x - before_x).abs() + (after_y - before_y).abs() >= 1;
            if moved {
                moving_frames += 1;
                if pose != AnimId::Fly {
                    moving_but_not_fly += 1;
                }
                if pose == AnimId::Fly {
                    fly_indices.insert(sprite.frame_idx());
                }
            }
        }

        assert!(moving_frames > 30, "bird barely moved ({moving_frames}); test setup is wrong");
        assert_eq!(
            moving_but_not_fly, 0,
            "bird MOVED on {moving_but_not_fly}/{moving_frames} frames while showing a non-fly pose (floats without flapping)"
        );
        assert!(
            fly_indices.len() >= 2,
            "fly animation never advanced past frame {fly_indices:?} while moving (frozen wings)"
        );
    }

    #[test]
    fn perched_bird_flits_and_returns() {
        let mut b = brain();
        let w = win(400, 300, 200, 150);
        let mut saw_flit = false;
        let mut saw_perch_after = false;
        // 16 s covers the worst-case flit delay; watch the sequence.
        for _ in 0..320 {
            b.update(0.05, Some(&w));
            if b.state == State::Flit {
                saw_flit = true;
            }
            if saw_flit && b.state == State::Perch {
                saw_perch_after = true;
            }
        }
        assert!(saw_flit, "bird never flitted");
        assert!(saw_perch_after, "bird never returned to perch after flitting");
    }
}
