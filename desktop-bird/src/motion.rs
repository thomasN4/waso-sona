//! Placeholder motion driver for the bird.
//!
//! This is a deliberately simple stand-in for the future shared `BirdBrain`
//! state machine. It just wanders: pick a random target inside the output,
//! ease toward it, occasionally idle, and report which way it is facing so the
//! sprite can be mirrored. The real perch / fly-to-window behaviour needs the
//! `WindowTracker` (next slice) and is intentionally out of scope here.

use rand::RngExt;

/// Travel speed in pixels per second.
const SPEED: f32 = 140.0;
/// Distance (px) at which we consider the target reached.
const ARRIVE_EPS: f32 = 2.0;

pub struct Wander {
    /// Current top-left position of the sprite, in buffer pixels.
    x: f32,
    y: f32,
    /// Current target top-left position.
    tx: f32,
    ty: f32,
    /// Inclusive upper bounds for the top-left so the sprite stays fully visible.
    max_x: f32,
    max_y: f32,
    /// Seconds left to sit still before moving again.
    idle: f32,
    /// True when the last horizontal movement was leftward.
    facing_left: bool,
    /// Whether bounds have been set (i.e. the surface has been configured).
    ready: bool,
}

impl Wander {
    pub fn new() -> Self {
        Wander {
            x: 0.0,
            y: 0.0,
            tx: 0.0,
            ty: 0.0,
            max_x: 0.0,
            max_y: 0.0,
            idle: 0.0,
            facing_left: false,
            ready: false,
        }
    }

    /// (Re)configure for a surface of `surf_w x surf_h` holding a
    /// `sprite_w x sprite_h` bird. Centres the bird and picks a first target.
    pub fn set_bounds(&mut self, surf_w: u32, surf_h: u32, sprite_w: u32, sprite_h: u32) {
        self.max_x = surf_w.saturating_sub(sprite_w).max(1) as f32;
        self.max_y = surf_h.saturating_sub(sprite_h).max(1) as f32;
        self.x = (self.max_x / 2.0).min(self.max_x);
        self.y = (self.max_y / 2.0).min(self.max_y);
        self.ready = true;
        self.pick_target();
    }

    fn pick_target(&mut self) {
        let mut rng = rand::rng();
        self.tx = rng.random_range(0.0..=self.max_x);
        self.ty = rng.random_range(0.0..=self.max_y);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if !self.ready {
            return;
        }
        if self.idle > 0.0 {
            self.idle -= dt;
            return;
        }

        let dx = self.tx - self.x;
        let dy = self.ty - self.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= ARRIVE_EPS {
            // Arrived: maybe perch a moment, then choose a new destination.
            let mut rng = rand::rng();
            if rng.random_bool(0.5) {
                self.idle = rng.random_range(0.4..1.6);
            }
            self.pick_target();
            return;
        }

        if dx.abs() > 0.5 {
            self.facing_left = dx < 0.0;
        }

        let step = SPEED * dt;
        if step >= dist {
            self.x = self.tx;
            self.y = self.ty;
        } else {
            self.x += dx / dist * step;
            self.y += dy / dist * step;
        }
    }

    /// Current top-left position, rounded to whole pixels.
    pub fn position(&self) -> (i32, i32) {
        (self.x.round() as i32, self.y.round() as i32)
    }

    pub fn facing_left(&self) -> bool {
        self.facing_left
    }
}
