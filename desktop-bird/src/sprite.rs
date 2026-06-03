//! Bird sprite: animation frames plus a procedural placeholder so the renderer
//! is demonstrable without any art assets yet (see RESEARCH.md open question #7).
//!
//! Real art can be dropped in later as a directory of PNG frames; set
//! `BIRD_SPRITE_DIR=/path/to/frames` and they are loaded instead of the
//! placeholder. Frames are sorted by filename and must all share one size.

use std::path::Path;

/// A single animation frame as straight (non-premultiplied) `RGBA8`.
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub pixels: Vec<u8>,
}

/// An animated sprite that faces **right** by default.
pub struct Sprite {
    frames: Vec<Frame>,
    idx: usize,
}

impl Sprite {
    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.idx]
    }

    /// Advance to the next animation frame (wraps around).
    pub fn advance(&mut self) {
        self.idx = (self.idx + 1) % self.frames.len();
    }

    /// Load PNG frames if `BIRD_SPRITE_DIR` is set and usable; otherwise `None`
    /// so the caller can fall back to [`Sprite::placeholder`].
    pub fn load_from_env() -> Option<Sprite> {
        let dir = std::env::var_os("BIRD_SPRITE_DIR")?;
        match Sprite::load_png_frames(Path::new(&dir)) {
            Ok(sprite) => Some(sprite),
            Err(err) => {
                eprintln!("desktop-bird: failed to load BIRD_SPRITE_DIR sprites: {err}; using placeholder");
                None
            }
        }
    }

    /// Load every `*.png` in `dir`, sorted by filename, as animation frames.
    pub fn load_png_frames(dir: &Path) -> Result<Sprite, String> {
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
        Ok(Sprite { frames, idx: 0 })
    }

    /// A tiny hand-drawn placeholder bird with a two-frame wing flap.
    pub fn placeholder() -> Sprite {
        Sprite { frames: vec![placeholder_frame(false), placeholder_frame(true)], idx: 0 }
    }
}

// --- placeholder art ------------------------------------------------------

const PW: u32 = 28;
const PH: u32 = 20;

// Straight RGBA colours.
const BODY: [u8; 4] = [58, 74, 107, 255]; // slate blue
const WING: [u8; 4] = [92, 112, 156, 255]; // lighter blue
const BEAK: [u8; 4] = [230, 150, 40, 255]; // orange
const EYE: [u8; 4] = [20, 20, 28, 255]; // near-black

/// Draw the placeholder bird; `wing_down` picks the flap pose.
fn placeholder_frame(wing_down: bool) -> Frame {
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

    // Wing: an oval over the body that lifts/drops between frames.
    let wing_cy = if wing_down { 14.0 } else { 8.0 };
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
