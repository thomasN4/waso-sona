//! Wire types shared by `desktop-bird` and `bird-panel`.
//!
//! The panel sends [`ControlMsg`]s to the running bird over a Unix socket as
//! newline-delimited JSON; the bird applies them live each frame. Keeping these
//! types in one crate that both binaries depend on means the protocol can never
//! drift between the two sides.
//!
//! Every `*Tuning` `Default` reproduces the bird's original hardcoded behaviour
//! exactly, so a freshly-launched bird and a freshly-launched panel agree on
//! the starting values with no handshake.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Where the bird listens and the panel connects:
/// `$XDG_RUNTIME_DIR/desktop-bird.sock`, falling back to the system temp dir
/// when the runtime dir isn't set. Both sides call this so they always agree.
pub fn socket_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    dir.join("desktop-bird.sock")
}

/// Movement / timing knobs — the old `brain.rs` constants.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionTuning {
    /// Travel speed, pixels/sec (was `SPEED`).
    pub speed: f32,
    /// Distance (px) at which a target counts as reached (was `ARRIVE_EPS`).
    pub arrive_eps: f32,
    /// While perched, follow a window move up to this far; beyond it, re-approach
    /// (was `FOLLOW_EPS`).
    pub follow_eps: f32,
    /// Seconds perched before a flit — random in `[min, max)` (was `FLIT_DELAY`).
    pub flit_delay_min: f32,
    pub flit_delay_max: f32,
    /// How far a flit strays from the perch (was `FLIT_RADIUS`).
    pub flit_radius: f32,
    /// Keep flapping this long after following a window move (was
    /// `FOLLOW_FLAP_LINGER`).
    pub follow_flap_linger: f32,
    /// Chance of pausing on a wander arrival (was `WANDER_IDLE_CHANCE`).
    pub wander_idle_chance: f64,
    /// Wander pause length, random in `[min, max)` (was `WANDER_IDLE`).
    pub wander_idle_min: f32,
    pub wander_idle_max: f32,
}

impl Default for MotionTuning {
    fn default() -> Self {
        MotionTuning {
            speed: 160.0,
            arrive_eps: 3.0,
            follow_eps: 48.0,
            flit_delay_min: 4.0,
            flit_delay_max: 10.0,
            flit_radius: 120.0,
            follow_flap_linger: 0.3,
            wander_idle_chance: 0.5,
            wander_idle_min: 0.4,
            wander_idle_max: 1.6,
        }
    }
}

/// Animation playback + procedural-pose amplitudes. The amplitude fields are
/// multipliers on the bird's base poses (1.0 = unchanged), so they make sense
/// across every style. They only affect the procedural bird; a `BIRD_SPRITE_DIR`
/// of custom PNGs can't be re-rendered, so only the `fps_*` fields apply there.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimTuning {
    pub fps_idle: f32,
    pub fps_fly: f32,
    pub fps_perch: f32,
    pub fps_talk: f32,
    /// Wing-sweep amplitude multiplier (scales the fly keyframes).
    pub wing_amp: f32,
    /// Idle body-settle multiplier.
    pub idle_bob: f32,
    /// Perch/talk body-settle multiplier.
    pub perch_bob: f32,
    /// Beak-open multiplier while chirping.
    pub talk_beak: f32,
}

impl Default for AnimTuning {
    fn default() -> Self {
        // Matches `Sprite::procedural` keyframes/fps exactly.
        AnimTuning {
            fps_idle: 3.0,
            fps_fly: 12.0,
            fps_perch: 2.0,
            fps_talk: 6.0,
            wing_amp: 1.0,
            idle_bob: 1.0,
            perch_bob: 1.0,
            talk_beak: 1.0,
        }
    }
}

/// Speech-bubble look — the old `render.rs` constants.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BubbleTuning {
    /// sitelen pona glyph height (was `TEXT_PX`).
    pub text_px: f32,
    /// Inner padding between text and border (was `PAD`).
    pub pad: i32,
    /// Triangular tail height (was `TAIL`).
    pub tail: i32,
    /// Corner rounding radius (was `RADIUS`).
    pub radius: i32,
    /// Text ink as a grey level 0..=255 (was `INK` = 0x22).
    pub ink: u32,
}

impl Default for BubbleTuning {
    fn default() -> Self {
        BubbleTuning { text_px: 22.0, pad: 6, tail: 6, radius: 5, ink: 0x22 }
    }
}

/// The full live-tunable state. `style` is carried separately via
/// [`ControlMsg::SetStyle`] so dragging a slider never silently overrides a bird
/// launched with a non-default `BIRD_STYLE`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tuning {
    pub motion: MotionTuning,
    pub anim: AnimTuning,
    pub bubble: BubbleTuning,
}

/// A behaviour state the panel can force for instant testing. Maps onto the
/// bird's internal state machine. Approach/Perch/Flit are only meaningful while
/// a window is focused (the bird perches on it); Wander always works.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForceState {
    Wander,
    Approach,
    Perch,
    Flit,
}

/// One command from the panel to the bird (newline-delimited JSON on the wire).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlMsg {
    /// Replace the live motion/anim/bubble tuning.
    SetTuning(Tuning),
    /// Switch the procedural bird's style by name (see `art::STYLES`).
    SetStyle(String),
    /// Pop a speech bubble right now (UCSUR sitelen pona text + seconds).
    Bubble { text: String, secs: f32 },
    /// Force a behaviour state.
    Force(ForceState),
}
