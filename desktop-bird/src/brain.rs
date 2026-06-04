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

use crate::render::Rect;
use crate::sprite::AnimId;
use crate::tracker::WindowInfo;

/// Travel speed, pixels/sec.
const SPEED: f32 = 160.0;
/// Distance (px) at which a target counts as reached.
const ARRIVE_EPS: f32 = 3.0;
/// While perched, a window move up to this far is followed smoothly; beyond it
/// we treat it as a new target and re-approach.
const FOLLOW_EPS: f32 = 48.0;
/// Seconds perched before taking a flit (random in this range).
const FLIT_DELAY: (f32, f32) = (4.0, 10.0);
/// How far a flit strays from the perch.
const FLIT_RADIUS: f32 = 120.0;
/// Wander: chance of pausing on arrival, and pause length (s).
const WANDER_IDLE_CHANCE: f64 = 0.5;
const WANDER_IDLE: (f32, f32) = (0.4, 1.6);

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
    facing_left: bool,
    /// Bounds set (surface configured) yet?
    ready: bool,
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
            facing_left: false,
            ready: false,
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

    /// Which animation clip the renderer should play.
    pub fn pose(&self) -> AnimId {
        match self.state {
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

    fn update_wander(&mut self, dt: f32) {
        if self.idle > 0.0 {
            self.idle -= dt;
            return;
        }
        if self.step_toward(dt) {
            let mut rng = rand::rng();
            if rng.random_bool(WANDER_IDLE_CHANCE) {
                self.idle = rng.random_range(WANDER_IDLE.0..WANDER_IDLE.1);
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
                if moved > FOLLOW_EPS {
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
                    self.flit_timer = rand::rng().random_range(FLIT_DELAY.0..FLIT_DELAY.1);
                }
            }
            State::Perch => {
                self.step_toward(dt); // ease along if the window is drifting
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
        // into the window).
        let dx = rng.random_range(-FLIT_RADIUS..FLIT_RADIUS);
        let dy = rng.random_range(-FLIT_RADIUS..FLIT_RADIUS / 3.0);
        self.tx = (perch.0 + dx).clamp(0.0, self.max_x);
        self.ty = (perch.1 + dy).clamp(0.0, self.max_y);
        self.state = State::Flit;
    }

    /// Move toward `(tx, ty)` at `SPEED`, updating facing. Returns whether the
    /// target was reached this step.
    fn step_toward(&mut self, dt: f32) -> bool {
        let dx = self.tx - self.x;
        let dy = self.ty - self.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dx.abs() > 0.5 {
            self.facing_left = dx < 0.0;
        }
        if dist <= ARRIVE_EPS {
            self.x = self.tx;
            self.y = self.ty;
            return true;
        }
        let step = SPEED * dt;
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
