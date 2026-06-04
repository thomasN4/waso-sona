//! Bird sprite: per-state animation clips plus procedural placeholders so the
//! renderer is demonstrable without art assets yet.
//!
//! The `BirdBrain` (`brain.rs`) picks an [`AnimId`] each frame; the `Sprite`
//! plays the matching clip. Real art drops in via `BIRD_SPRITE_DIR`:
//!   - per-state layout: `idle/`, `fly/`, `perch/` subdirs, each a set of PNGs
//!     sorted by filename (this is where the upcoming vector poses go);
//!   - or a flat directory of PNGs, used as a single clip for every state.
//!
//! All frames must share one canvas size.

use std::collections::HashMap;
use std::path::Path;

/// Which animation to play; selected by `BirdBrain::pose`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AnimId {
    Idle,
    Fly,
    Perch,
}

impl AnimId {
    const ALL: [AnimId; 3] = [AnimId::Idle, AnimId::Fly, AnimId::Perch];

    fn subdir(self) -> &'static str {
        match self {
            AnimId::Idle => "idle",
            AnimId::Fly => "fly",
            AnimId::Perch => "perch",
        }
    }

    /// Default playback rate (frames/sec) for the procedural + flat loaders.
    fn default_fps(self) -> f32 {
        match self {
            AnimId::Idle => 3.0,
            AnimId::Fly => 8.0,
            AnimId::Perch => 2.0,
        }
    }
}

/// A single animation frame as straight (non-premultiplied) `RGBA8`.
#[derive(Clone)]
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub pixels: Vec<u8>,
}

/// One looping animation clip.
#[derive(Clone)]
struct Clip {
    frames: Vec<Frame>,
    fps: f32,
}

/// An animated sprite (faces **right** by default) with playback state.
pub struct Sprite {
    clips: HashMap<AnimId, Clip>, // always fully populated (missing ones aliased)
    current: AnimId,
    idx: usize,
    accum: f32,
}

impl Sprite {
    /// Build from three clips, populating all `AnimId`s.
    fn from_clips(idle: Clip, fly: Clip, perch: Clip) -> Sprite {
        let mut clips = HashMap::new();
        clips.insert(AnimId::Idle, idle);
        clips.insert(AnimId::Fly, fly);
        clips.insert(AnimId::Perch, perch);
        Sprite { clips, current: AnimId::Idle, idx: 0, accum: 0.0 }
    }

    /// Switch to a clip, restarting it if it actually changed.
    pub fn set_anim(&mut self, id: AnimId) {
        if id != self.current {
            self.current = id;
            self.idx = 0;
            self.accum = 0.0;
        }
    }

    /// Advance playback by `dt` seconds using the current clip's frame rate.
    pub fn advance(&mut self, dt: f32) {
        let clip = &self.clips[&self.current];
        if clip.frames.len() <= 1 || clip.fps <= 0.0 {
            return;
        }
        let frame_time = 1.0 / clip.fps;
        self.accum += dt;
        while self.accum >= frame_time {
            self.accum -= frame_time;
            self.idx = (self.idx + 1) % clip.frames.len();
        }
    }

    pub fn current_frame(&self) -> &Frame {
        let clip = &self.clips[&self.current];
        &clip.frames[self.idx.min(clip.frames.len() - 1)]
    }

    /// Canvas size shared by all frames (used for bounds/perch math).
    pub fn frame_size(&self) -> (u32, u32) {
        let f = &self.clips[&AnimId::Idle].frames[0];
        (f.w, f.h)
    }

    /// Load clips if `BIRD_SPRITE_DIR` is set and usable, else `None` so the
    /// caller falls back to [`Sprite::placeholder`].
    pub fn load_from_env() -> Option<Sprite> {
        let dir = std::env::var_os("BIRD_SPRITE_DIR")?;
        match Sprite::load_dir(Path::new(&dir)) {
            Ok(sprite) => Some(sprite),
            Err(err) => {
                eprintln!("desktop-bird: failed to load BIRD_SPRITE_DIR: {err}; using placeholder");
                None
            }
        }
    }

    /// Load from a directory: per-state subdirs if present, else a flat dir as a
    /// single clip used for every state.
    pub fn load_dir(dir: &Path) -> Result<Sprite, String> {
        let has_subdirs = AnimId::ALL.iter().any(|a| dir.join(a.subdir()).is_dir());

        if has_subdirs {
            // Load whichever subdirs exist; alias missing ones to a present clip.
            let load = |id: AnimId| -> Option<Clip> {
                let p = dir.join(id.subdir());
                p.is_dir().then(|| load_clip(&p, id.default_fps())).and_then(Result::ok)
            };
            let idle = load(AnimId::Idle);
            let fly = load(AnimId::Fly);
            let perch = load(AnimId::Perch);

            let idle_c = idle.clone().or_else(|| fly.clone()).or_else(|| perch.clone());
            let idle_c = idle_c.ok_or_else(|| format!("no usable clip subdirs in {}", dir.display()))?;
            let fly_c = fly.unwrap_or_else(|| idle_c.clone());
            let perch_c = perch.unwrap_or_else(|| idle_c.clone());
            Ok(Sprite::from_clips(idle_c, fly_c, perch_c))
        } else {
            let clip = load_clip(dir, AnimId::Fly.default_fps())?;
            Ok(Sprite::from_clips(clip.clone(), clip.clone(), clip))
        }
    }

    /// A tiny hand-drawn placeholder bird with distinct idle / fly / perch clips.
    pub fn placeholder() -> Sprite {
        // wing centre-y per pose; bigger swing = flapping.
        let fly = Clip { frames: vec![bird_frame(8.0), bird_frame(14.0)], fps: 8.0 };
        let idle = Clip { frames: vec![bird_frame(11.5), bird_frame(12.5)], fps: 3.0 };
        let perch = Clip { frames: vec![bird_frame(12.5)], fps: 2.0 };
        Sprite::from_clips(idle, fly, perch)
    }
}

/// Load a directory of PNGs (sorted by filename) into a clip.
fn load_clip(dir: &Path, fps: f32) -> Result<Clip, String> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .png frames in {}", dir.display()));
    }
    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        let img = image::open(&path)
            .map_err(|e| format!("decode {}: {e}", path.display()))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        frames.push(Frame { w, h, pixels: img.into_raw() });
    }
    Ok(Clip { frames, fps })
}

// --- placeholder art ------------------------------------------------------

const PW: u32 = 28;
const PH: u32 = 20;

// Straight RGBA colours.
const BODY: [u8; 4] = [58, 74, 107, 255]; // slate blue
const WING: [u8; 4] = [92, 112, 156, 255]; // lighter blue
const BEAK: [u8; 4] = [230, 150, 40, 255]; // orange
const EYE: [u8; 4] = [20, 20, 28, 255]; // near-black

/// Draw the placeholder bird with the wing oval centred at `wing_cy` (lets us
/// build flap / folded poses from one routine).
fn bird_frame(wing_cy: f32) -> Frame {
    let mut px = vec![0u8; (PW * PH * 4) as usize];
    let mut put = |x: i32, y: i32, c: [u8; 4]| {
        if x >= 0 && y >= 0 && (x as u32) < PW && (y as u32) < PH {
            let i = ((y as u32 * PW + x as u32) * 4) as usize;
            px[i..i + 4].copy_from_slice(&c);
        }
    };
    let ellipse = |cx: f32, cy: f32, rx: f32, ry: f32, c: [u8; 4], put: &mut dyn FnMut(i32, i32, [u8; 4])| {
        for y in 0..PH as i32 {
            for x in 0..PW as i32 {
                let nx = (x as f32 + 0.5 - cx) / rx;
                let ny = (y as f32 + 0.5 - cy) / ry;
                if nx * nx + ny * ny <= 1.0 {
                    put(x, y, c);
                }
            }
        }
    };

    // Body (oval) and head (circle), facing right.
    ellipse(11.0, 11.0, 8.0, 6.0, BODY, &mut put);
    ellipse(20.0, 8.0, 4.5, 4.5, BODY, &mut put);

    // Tail wedge on the left.
    for y in 8i32..14 {
        for x in 0i32..4 {
            if (y - 11).abs() <= x {
                put(x, y, BODY);
            }
        }
    }

    // Wing: an oval over the body whose height selects the pose.
    ellipse(10.0, wing_cy, 5.0, 2.5, WING, &mut put);

    // Beak (small triangle) and eye on the head.
    for d in 0..3 {
        for y in (7 - d)..=(7 + d) {
            put(24 + d, y, BEAK);
        }
    }
    put(21, 7, EYE);

    Frame { w: PW, h: PH, pixels: px }
}
