//! Bird sprite: per-state animation clips plus a procedural bird drawn as
//! anti-aliased vector shapes in code (no art assets needed).
//!
//! The `BirdBrain` (`brain.rs`) picks an [`AnimId`] each frame; the `Sprite`
//! plays the matching clip. The procedural bird comes in several colour
//! schemes — pick one with `BIRD_STYLE` (see [`crate::art::STYLES`]). External art can
//! still replace it via `BIRD_SPRITE_DIR`:
//!   - per-state layout: `idle/`, `fly/`, `perch/`, `talk/` subdirs, each a set
//!     of PNGs sorted by filename;
//!   - or a flat directory of PNGs, used as a single clip for every state.
//!
//! All frames must share one canvas size.

use crate::art::{style_from_env, BirdStyle, Pose};
use bird_protocol::AnimTuning;
use std::collections::HashMap;
use std::path::Path;

/// Which animation to play; selected by `BirdBrain::pose`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AnimId {
    Idle,
    Fly,
    Perch,
    /// Played while a speech bubble is showing (beak opens/closes — chirping).
    Talk,
}

impl AnimId {
    const ALL: [AnimId; 4] = [AnimId::Idle, AnimId::Fly, AnimId::Perch, AnimId::Talk];

    fn subdir(self) -> &'static str {
        match self {
            AnimId::Idle => "idle",
            AnimId::Fly => "fly",
            AnimId::Perch => "perch",
            AnimId::Talk => "talk",
        }
    }

    /// Default playback rate (frames/sec) for the procedural + flat loaders.
    fn default_fps(self) -> f32 {
        match self {
            AnimId::Idle => 3.0,
            AnimId::Fly => 8.0,
            AnimId::Perch => 2.0,
            AnimId::Talk => 6.0,
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
    /// Build from the four clips, populating all `AnimId`s.
    fn from_clips(idle: Clip, fly: Clip, perch: Clip, talk: Clip) -> Sprite {
        let mut clips = HashMap::new();
        clips.insert(AnimId::Idle, idle);
        clips.insert(AnimId::Fly, fly);
        clips.insert(AnimId::Perch, perch);
        clips.insert(AnimId::Talk, talk);
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

    /// The animation currently playing — used to restore it after a live rebuild.
    pub fn current_anim(&self) -> AnimId {
        self.current
    }

    /// Live-set one clip's playback rate (control panel). Works for procedural
    /// and custom-PNG sprites alike, no rebuild needed.
    pub fn set_fps(&mut self, id: AnimId, fps: f32) {
        if let Some(clip) = self.clips.get_mut(&id) {
            clip.fps = fps;
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

    /// Current frame index within the playing clip (used by tests + BIRD_DEBUG).
    pub fn frame_idx(&self) -> usize {
        self.idx
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

            let talk = load(AnimId::Talk);

            let idle_c = idle.clone().or_else(|| fly.clone()).or_else(|| perch.clone());
            let idle_c = idle_c.ok_or_else(|| format!("no usable clip subdirs in {}", dir.display()))?;
            let fly_c = fly.unwrap_or_else(|| idle_c.clone());
            let perch_c = perch.unwrap_or_else(|| idle_c.clone());
            let talk_c = talk.unwrap_or_else(|| idle_c.clone());
            Ok(Sprite::from_clips(idle_c, fly_c, perch_c, talk_c))
        } else {
            let clip = load_clip(dir, AnimId::Fly.default_fps())?;
            Ok(Sprite::from_clips(clip.clone(), clip.clone(), clip.clone(), clip))
        }
    }

    /// Write every clip's frames as PNGs under `dir/<anim>/NN.png` — the same
    /// per-state layout [`load_dir`] reads back. Lets the code-generated
    /// placeholder be snapshotted to disk and later swapped for real art.
    pub fn write_png_frames(&self, dir: &Path) -> Result<(), String> {
        for id in AnimId::ALL {
            let sub = dir.join(id.subdir());
            std::fs::create_dir_all(&sub).map_err(|e| format!("create {}: {e}", sub.display()))?;
            for (i, frame) in self.clips[&id].frames.iter().enumerate() {
                let path = sub.join(format!("{i:02}.png"));
                let img = image::RgbaImage::from_raw(frame.w, frame.h, frame.pixels.clone())
                    .ok_or_else(|| format!("bad frame buffer for {}", path.display()))?;
                img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;
            }
        }
        Ok(())
    }

    /// The procedural bird with distinct idle / fly / perch / talk clips, in
    /// the style selected by `BIRD_STYLE` (default: the first of [`crate::art::STYLES`]).
    pub fn placeholder() -> Sprite {
        Sprite::procedural(style_from_env())
    }

    /// Build the procedural bird in a given style with the default animation.
    pub fn procedural(st: &BirdStyle) -> Sprite {
        Sprite::procedural_tuned(st, &AnimTuning::default())
    }

    /// Build the procedural bird in a given style, with the pose amplitudes and
    /// frame rates scaled by `a` (the control panel's live animation tuning).
    /// `AnimTuning::default()` reproduces the original keyframes exactly.
    pub fn procedural_tuned(st: &BirdStyle, a: &AnimTuning) -> Sprite {
        // Fly: a ping-pong flap (up / mid / down / mid) so the wing sweeps
        // smoothly instead of toggling between extremes. `wing_amp` scales the
        // whole sweep.
        let w = a.wing_amp;
        let fly = Clip {
            frames: vec![
                st.frame(&Pose { wing: 0.55 * w, ..Pose::FLYING }),
                st.frame(&Pose { wing: -0.05 * w, ..Pose::FLYING }),
                st.frame(&Pose { wing: -0.65 * w, ..Pose::FLYING }),
                st.frame(&Pose { wing: -0.05 * w, ..Pose::FLYING }),
            ],
            fps: a.fps_fly,
        };
        // Idle: standing, the body gently settling between two heights.
        let idle = Clip {
            frames: vec![
                st.frame(&Pose::STANDING),
                st.frame(&Pose { bob: 1.0 * a.idle_bob, ..Pose::STANDING }),
            ],
            fps: a.fps_idle,
        };
        let perch_bob = 0.6 * a.perch_bob;
        let perch = Clip { frames: vec![st.frame(&Pose { bob: perch_bob, ..Pose::STANDING })], fps: a.fps_perch };
        // Talk: perched, beak opening and closing.
        let talk = Clip {
            frames: vec![
                st.frame(&Pose { bob: perch_bob, ..Pose::STANDING }),
                st.frame(&Pose { beak: 1.0 * a.talk_beak, bob: perch_bob, ..Pose::STANDING }),
            ],
            fps: a.fps_talk,
        };
        Sprite::from_clips(idle, fly, perch, talk)
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

