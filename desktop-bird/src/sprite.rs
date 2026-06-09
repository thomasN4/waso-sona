//! Bird sprite: per-state animation clips plus a procedural bird drawn as
//! anti-aliased vector shapes in code (no art assets needed).
//!
//! The `BirdBrain` (`brain.rs`) picks an [`AnimId`] each frame; the `Sprite`
//! plays the matching clip. The procedural bird comes in several colour
//! schemes — pick one with `BIRD_STYLE` (see [`STYLES`]). External art can
//! still replace it via `BIRD_SPRITE_DIR`:
//!   - per-state layout: `idle/`, `fly/`, `perch/`, `talk/` subdirs, each a set
//!     of PNGs sorted by filename;
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
    /// the style selected by `BIRD_STYLE` (default: the first of [`STYLES`]).
    pub fn placeholder() -> Sprite {
        Sprite::procedural(style_from_env())
    }

    /// Build the procedural bird in a given style.
    pub fn procedural(st: &BirdStyle) -> Sprite {
        // Fly: a ping-pong flap (up / mid / down / mid) so the wing sweeps
        // smoothly instead of toggling between extremes.
        let fly = Clip {
            frames: vec![
                bird_frame(st, &Pose { wing: 0.55, ..Pose::FLYING }),
                bird_frame(st, &Pose { wing: -0.05, ..Pose::FLYING }),
                bird_frame(st, &Pose { wing: -0.65, ..Pose::FLYING }),
                bird_frame(st, &Pose { wing: -0.05, ..Pose::FLYING }),
            ],
            fps: 12.0,
        };
        // Idle: standing, the body gently settling between two heights.
        let idle = Clip {
            frames: vec![
                bird_frame(st, &Pose::STANDING),
                bird_frame(st, &Pose { bob: 1.0, ..Pose::STANDING }),
            ],
            fps: 3.0,
        };
        let perch = Clip { frames: vec![bird_frame(st, &Pose { bob: 0.6, ..Pose::STANDING })], fps: 2.0 };
        // Talk: perched, beak opening and closing.
        let talk = Clip {
            frames: vec![
                bird_frame(st, &Pose { bob: 0.6, ..Pose::STANDING }),
                bird_frame(st, &Pose { beak: 1.0, bob: 0.6, ..Pose::STANDING }),
            ],
            fps: 6.0,
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

// --- procedural art ---------------------------------------------------------
//
// The bird is "vector" art authored in code: analytic shapes (rotated
// ellipses, capsules, triangles) composited back-to-front into a frame with
// coverage-based anti-aliasing. One drawing routine renders every pose; a
// [`BirdStyle`] supplies the palette and optional plumage markings.

const PW: u32 = 56;
const PH: u32 = 40;
/// Coverage supersamples per axis (SS x SS samples per pixel).
const SS: i32 = 4;
/// Extra radius of the dark underlay behind each silhouette shape — a soft
/// rim so the bird reads on any desktop background.
const RIM: f32 = 0.8;

type Rgba = [u8; 4];

const EYE_WHITE: Rgba = [250, 250, 248, 255];
const EYE_DARK: Rgba = [24, 24, 30, 255];

/// One colour-and-marking scheme for the procedural bird. All presets share
/// the same geometry; only palette and optional markings differ.
pub struct BirdStyle {
    pub name: &'static str,
    body: Rgba,
    wing: Rgba,
    belly: Rgba,
    beak: Rgba,
    feet: Rgba,
    cap: Option<Rgba>,
    cheek: Option<Rgba>,
    bib: Option<Rgba>,
    breast: Option<Rgba>,
    wing_bar: Option<Rgba>,
}

/// The built-in styles; pick one by name (case-insensitive) via `BIRD_STYLE`.
pub const STYLES: [BirdStyle; 5] = [
    // Slate blue with a cream belly — the original bird's identity, refined.
    BirdStyle {
        name: "sona",
        body: [84, 104, 148, 255],
        wing: [122, 142, 184, 255],
        belly: [236, 228, 205, 255],
        beak: [232, 156, 46, 255],
        feet: [196, 130, 62, 255],
        cap: None,
        cheek: None,
        bib: None,
        breast: None,
        wing_bar: None,
    },
    // Chickadee: grey back, black cap and bib, white cheek and belly.
    BirdStyle {
        name: "walo",
        body: [146, 154, 164, 255],
        wing: [112, 122, 134, 255],
        belly: [242, 240, 232, 255],
        beak: [52, 52, 58, 255],
        feet: [96, 96, 104, 255],
        cap: Some([38, 40, 46, 255]),
        cheek: Some([248, 248, 246, 255]),
        bib: Some([38, 40, 46, 255]),
        breast: None,
        wing_bar: None,
    },
    // Bluebird: vivid blue back and head, rusty chest, white belly.
    BirdStyle {
        name: "laso",
        body: [58, 118, 196, 255],
        wing: [42, 90, 158, 255],
        belly: [246, 243, 236, 255],
        beak: [62, 62, 70, 255],
        feet: [122, 102, 92, 255],
        cap: None,
        cheek: None,
        bib: None,
        breast: Some([214, 122, 62, 255]),
        wing_bar: None,
    },
    // Goldfinch: yellow body, black cap, dark wing with a light bar.
    BirdStyle {
        name: "jelo",
        body: [238, 196, 48, 255],
        wing: [56, 58, 64, 255],
        belly: [250, 242, 200, 255],
        beak: [232, 168, 92, 255],
        feet: [182, 142, 102, 255],
        cap: Some([40, 42, 48, 255]),
        cheek: None,
        bib: None,
        breast: None,
        wing_bar: Some([248, 246, 238, 255]),
    },
    // Robin: brown-grey back, red-orange breast, off-white belly.
    BirdStyle {
        name: "loje",
        body: [112, 98, 88, 255],
        wing: [94, 82, 74, 255],
        belly: [240, 235, 225, 255],
        beak: [228, 178, 64, 255],
        feet: [122, 102, 90, 255],
        cap: None,
        cheek: None,
        bib: None,
        breast: Some([226, 104, 58, 255]),
        wing_bar: None,
    },
];

/// Resolve `BIRD_STYLE` against [`STYLES`], defaulting to the first preset.
pub fn style_from_env() -> &'static BirdStyle {
    let Some(want) = std::env::var_os("BIRD_STYLE") else { return &STYLES[0] };
    let want = want.to_string_lossy();
    STYLES.iter().find(|s| s.name.eq_ignore_ascii_case(&want)).unwrap_or_else(|| {
        let names: Vec<_> = STYLES.iter().map(|s| s.name).collect();
        eprintln!(
            "desktop-bird: unknown BIRD_STYLE {want:?} (have: {}); using {}",
            names.join(", "),
            STYLES[0].name
        );
        &STYLES[0]
    })
}

/// One body configuration of the bird; each animation frame renders one pose.
#[derive(Clone, Copy)]
struct Pose {
    /// Wing sweep in radians: 0 = trailing along the body, positive = raised.
    wing: f32,
    /// Beak opening, 0.0 (closed) ..= 1.0 (wide).
    beak: f32,
    /// Vertical settle of the body in pixels; the feet stay planted.
    bob: f32,
    /// Legs drawn (standing / perched) or tucked away (flight).
    feet: bool,
}

impl Pose {
    const FLYING: Pose = Pose { wing: 0.0, beak: 0.0, bob: 0.0, feet: false };
    const STANDING: Pose = Pose { wing: -0.12, beak: 0.0, bob: 0.0, feet: true };
}

/// Scale a colour's channels by `f` (alpha unchanged) — shading and rims.
fn shade(c: Rgba, f: f32) -> Rgba {
    let s = |v: u8| (v as f32 * f).round().clamp(0.0, 255.0) as u8;
    [s(c[0]), s(c[1]), s(c[2]), c[3]]
}

/// Premultiplied-RGBA float canvas the bird's shapes composite into.
struct Canvas {
    px: Vec<f32>,
}

impl Canvas {
    fn new() -> Canvas {
        Canvas { px: vec![0.0; (PW * PH * 4) as usize] }
    }

    /// Source-over `color` wherever `inside` holds, anti-aliased by SS x SS
    /// coverage sampling per pixel.
    fn fill(&mut self, color: Rgba, inside: &dyn Fn(f32, f32) -> bool) {
        let ca = color[3] as f32 / 255.0;
        let src = [
            color[0] as f32 / 255.0 * ca,
            color[1] as f32 / 255.0 * ca,
            color[2] as f32 / 255.0 * ca,
            ca,
        ];
        for y in 0..PH as i32 {
            for x in 0..PW as i32 {
                let mut hits = 0;
                for sy in 0..SS {
                    for sx in 0..SS {
                        let fx = x as f32 + (sx as f32 + 0.5) / SS as f32;
                        let fy = y as f32 + (sy as f32 + 0.5) / SS as f32;
                        if inside(fx, fy) {
                            hits += 1;
                        }
                    }
                }
                if hits == 0 {
                    continue;
                }
                let cov = hits as f32 / (SS * SS) as f32;
                let inv = 1.0 - ca * cov;
                let i = ((y * PW as i32 + x) * 4) as usize;
                for (dst, s) in self.px[i..i + 4].iter_mut().zip(src) {
                    *dst = s * cov + *dst * inv;
                }
            }
        }
    }

    /// Flatten to a straight-RGBA [`Frame`].
    fn frame(self) -> Frame {
        let mut pixels = vec![0u8; (PW * PH * 4) as usize];
        for (i, p) in self.px.chunks_exact(4).enumerate() {
            let a = p[3];
            if a <= 0.0 {
                continue;
            }
            let o = i * 4;
            let q = |v: f32| (v / a * 255.0).round().clamp(0.0, 255.0) as u8;
            pixels[o] = q(p[0]);
            pixels[o + 1] = q(p[1]);
            pixels[o + 2] = q(p[2]);
            pixels[o + 3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        Frame { w: PW, h: PH, pixels }
    }
}

// Shape inside-tests, in pixel coordinates.

fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, rot: f32) -> impl Fn(f32, f32) -> bool {
    let (s, c) = rot.sin_cos();
    move |x: f32, y: f32| {
        let (dx, dy) = (x - cx, y - cy);
        let u = (dx * c + dy * s) / rx;
        let v = (dy * c - dx * s) / ry;
        u * u + v * v <= 1.0
    }
}

fn circle(cx: f32, cy: f32, r: f32) -> impl Fn(f32, f32) -> bool {
    ellipse(cx, cy, r, r, 0.0)
}

/// A line segment with round caps and thickness `2r`.
fn capsule(p: (f32, f32), q: (f32, f32), r: f32) -> impl Fn(f32, f32) -> bool {
    let (vx, vy) = (q.0 - p.0, q.1 - p.1);
    let len2 = (vx * vx + vy * vy).max(1e-6);
    move |x: f32, y: f32| {
        let t = (((x - p.0) * vx + (y - p.1) * vy) / len2).clamp(0.0, 1.0);
        let (dx, dy) = (x - (p.0 + t * vx), y - (p.1 + t * vy));
        dx * dx + dy * dy <= r * r
    }
}

fn tri(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> impl Fn(f32, f32) -> bool {
    move |x: f32, y: f32| {
        let edge = |p: (f32, f32), q: (f32, f32)| (x - q.0) * (p.1 - q.1) - (p.0 - q.0) * (y - q.1);
        let (d1, d2, d3) = (edge(a, b), edge(b, c), edge(c, a));
        !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
    }
}

/// Push a triangle's vertices out from its centroid by `d` pixels (rim
/// underlays for the beak).
fn grow_tri(t: [(f32, f32); 3], d: f32) -> [(f32, f32); 3] {
    let cx = (t[0].0 + t[1].0 + t[2].0) / 3.0;
    let cy = (t[0].1 + t[1].1 + t[2].1) / 3.0;
    t.map(|p| {
        let (vx, vy) = (p.0 - cx, p.1 - cy);
        let l = (vx * vx + vy * vy).sqrt().max(1e-3);
        (p.0 + vx / l * d, p.1 + vy / l * d)
    })
}

/// Draw one frame of the bird (facing right) in the given style and pose.
fn bird_frame(st: &BirdStyle, pose: &Pose) -> Frame {
    let mut c = Canvas::new();
    let b = pose.bob;
    let outline = shade(st.body, 0.45);

    // Torso silhouette: a tilted body oval, a round head, and a neck oval
    // blending the two.
    let body = ellipse(24.0, 24.0 + b, 13.0, 9.5, -0.12);
    let neck = ellipse(31.5, 17.5 + b, 8.0, 7.5, 0.0);
    let head = circle(37.5, 13.5 + b, 8.2);
    let torso = |x: f32, y: f32| body(x, y) || neck(x, y) || head(x, y);

    // Tail: three feather capsules fanning back from the rump, tilting against
    // the wing stroke in flight. Per-feather rims double as separations.
    let root = (14.0, 21.5 + b);
    let (ts, tc) = (-pose.wing * 0.3).sin_cos();
    for (tx, ty) in [(3.5, 16.5), (2.5, 21.0), (4.0, 25.5)] {
        let (dx, dy) = (tx - root.0, ty + b - root.1);
        let tip = (root.0 + dx * tc - dy * ts, root.1 + dx * ts + dy * tc);
        c.fill(outline, &capsule(root, tip, 2.0 + RIM));
        c.fill(shade(st.body, 0.85), &capsule(root, tip, 2.0));
    }

    // Legs, drawn before the body so they emerge from under the belly: a shank
    // to the ankle, then a front and a back toe.
    if pose.feet {
        for lx in [22.5_f32, 28.5] {
            let ankle = (lx + 0.7, 36.4);
            let segs = [
                ((lx, 30.0 + b), ankle, 1.2),
                (ankle, (ankle.0 + 2.8, 37.6), 0.9),
                (ankle, (ankle.0 - 2.2, 37.5), 0.9),
            ];
            for (p, q, r) in segs {
                c.fill(shade(st.feet, 0.5), &capsule(p, q, r + 0.6));
            }
            for (p, q, r) in segs {
                c.fill(st.feet, &capsule(p, q, r));
            }
        }
    }

    // Torso rim then fill. The two underlays merge into one slightly larger
    // silhouette (the neck sits inside their union), so the rim only shows at
    // the outer edge.
    c.fill(outline, &ellipse(24.0, 24.0 + b, 13.0 + RIM, 9.5 + RIM, -0.12));
    c.fill(outline, &circle(37.5, 13.5 + b, 8.2 + RIM));
    c.fill(st.body, &torso);

    // Plumage, clipped to the torso: breast patch, then belly over its lower
    // half, then head markings.
    if let Some(col) = st.breast {
        let patch = ellipse(33.0, 22.0 + b, 8.5, 8.0, 0.35);
        c.fill(col, &|x, y| patch(x, y) && torso(x, y));
    }
    let belly = ellipse(27.0, 29.0 + b, 10.0, 6.5, 0.18);
    c.fill(st.belly, &|x, y| belly(x, y) && torso(x, y));
    if let Some(col) = st.cap {
        c.fill(col, &|x, y| head(x, y) && y < 12.0 + b);
    }
    if let Some(col) = st.cheek {
        let patch = ellipse(39.5, 16.0 + b, 5.5, 4.8, -0.15);
        c.fill(col, &|x, y| patch(x, y) && head(x, y));
    }
    if let Some(col) = st.bib {
        let patch = ellipse(41.0, 20.0 + b, 4.5, 3.5, 0.5);
        c.fill(col, &|x, y| patch(x, y) && torso(x, y));
    }

    // Wing: an oval hung from the shoulder, swept by the pose angle. Feather
    // slits near the tip and an optional bar give it depth.
    let sh = (27.0, 19.0 + b);
    let dir = (-pose.wing.cos(), -pose.wing.sin());
    let perp = (dir.1, -dir.0); // unit, toward the wing's trailing edge
    let wc = (sh.0 + dir.0 * 6.5, sh.1 + dir.1 * 6.5);
    let rot = dir.1.atan2(dir.0);
    let wing = ellipse(wc.0, wc.1, 10.5, 4.8, rot);
    c.fill(shade(st.wing, 0.55), &ellipse(wc.0, wc.1, 10.5 + 0.7, 4.8 + 0.7, rot));
    c.fill(st.wing, &wing);
    for (along, out) in [(4.0, 1.2), (6.8, 2.2)] {
        let p = (wc.0 + dir.0 * along + perp.0 * out, wc.1 + dir.1 * along + perp.1 * out);
        let q = (p.0 + dir.0 * 5.5, p.1 + dir.1 * 5.5);
        let slit = capsule(p, q, 0.9);
        c.fill(shade(st.wing, 0.72), &|x, y| slit(x, y) && wing(x, y));
    }
    if let Some(col) = st.wing_bar {
        let p = (wc.0 + dir.0 * 0.5 - perp.0 * 3.5, wc.1 + dir.1 * 0.5 - perp.1 * 3.5);
        let q = (wc.0 + dir.0 * 2.0 + perp.0 * 3.5, wc.1 + dir.1 * 2.0 + perp.1 * 3.5);
        let bar = capsule(p, q, 1.3);
        c.fill(col, &|x, y| bar(x, y) && wing(x, y));
    }

    // Beak: upper and lower mandible triangles whose tips swing apart as the
    // beak opens. The lower one is shaded darker.
    let t = pose.beak;
    let upper = [(44.0, 11.8 + b), (44.0, 14.3 + b), (52.8 - t, 13.8 + b - 3.0 * t)];
    let lower = [(44.0, 13.9 + b), (44.0, 16.2 + b), (52.3 - 0.8 * t, 14.5 + b + 3.2 * t)];
    for (m, col) in [(lower, shade(st.beak, 0.72)), (upper, st.beak)] {
        let g = grow_tri(m, 0.7);
        c.fill(shade(st.beak, 0.5), &tri(g[0], g[1], g[2]));
        c.fill(col, &tri(m[0], m[1], m[2]));
    }

    // Cartoon eye: shaded ring, white sclera, dark iris, tiny highlight.
    c.fill(shade(st.body, 0.45), &circle(40.0, 12.6 + b, 3.4));
    c.fill(EYE_WHITE, &circle(40.0, 12.6 + b, 3.0));
    c.fill(EYE_DARK, &circle(40.6, 12.9 + b, 2.0));
    c.fill(EYE_WHITE, &circle(41.3, 12.1 + b, 0.7));

    c.frame()
}

/// Render every [`STYLES`] preset across key poses into one contact-sheet PNG
/// (rows = styles in declaration order; columns = perch, wing up, wing down,
/// talk, idle settle), scaled 4x nearest for inspection. No Wayland needed.
pub fn write_style_sheet(path: &Path) -> Result<(), String> {
    const SCALE: u32 = 4;
    const PAD: u32 = 8;
    let poses = [
        Pose { bob: 0.6, ..Pose::STANDING },
        Pose { wing: 0.55, ..Pose::FLYING },
        Pose { wing: -0.65, ..Pose::FLYING },
        Pose { beak: 1.0, bob: 0.6, ..Pose::STANDING },
        Pose { bob: 1.0, ..Pose::STANDING },
    ];
    let cell_w = PW * SCALE + PAD;
    let cell_h = PH * SCALE + PAD;
    let sheet_w = PAD + cell_w * poses.len() as u32;
    let sheet_h = PAD + cell_h * STYLES.len() as u32;
    let mut img = image::RgbaImage::from_pixel(sheet_w, sheet_h, image::Rgba([226, 228, 232, 255]));
    for (row, st) in STYLES.iter().enumerate() {
        for (col, pose) in poses.iter().enumerate() {
            let f = bird_frame(st, pose);
            let ox = PAD + col as u32 * cell_w;
            let oy = PAD + row as u32 * cell_h;
            for y in 0..PH * SCALE {
                for x in 0..PW * SCALE {
                    let i = (((y / SCALE) * PW + x / SCALE) * 4) as usize;
                    let a = f.pixels[i + 3] as u32;
                    if a == 0 {
                        continue;
                    }
                    let px = img.get_pixel_mut(ox + x, oy + y);
                    for ch in 0..3 {
                        let s = f.pixels[i + ch] as u32;
                        px.0[ch] = ((s * a + px.0[ch] as u32 * (255 - a)) / 255) as u8;
                    }
                }
            }
        }
    }
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}
