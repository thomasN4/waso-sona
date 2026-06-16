//! Procedural bird art: anti-aliased analytic shapes (rotated ellipses,
//! capsules, triangles) composited back-to-front into a [`Frame`].
//!
//! Every entry in [`STYLES`] names a `draw` routine plus a palette. The first
//! five share the cartoon geometry and differ only in plumage; the rest are
//! distinct looks (round chick, retro pixel sprite, origami, ink doodle,
//! swallow). All render the same [`Pose`]s on one 56x40 canvas, facing right.

use crate::sprite::Frame;
use std::path::Path;

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

/// One bird look: a drawing routine plus the palette it works from. How the
/// palette fields are interpreted is up to the routine (e.g. the doodle uses
/// `wing` as its ink colour).
pub struct BirdStyle {
    pub name: &'static str,
    draw: fn(&BirdStyle, &Pose) -> Frame,
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

impl BirdStyle {
    /// Render one animation frame of this bird in the given pose.
    pub fn frame(&self, pose: &Pose) -> Frame {
        (self.draw)(self, pose)
    }
}

/// No optional markings — for styles that don't use them.
const PLAIN: BirdStyle = BirdStyle {
    name: "",
    draw: draw_cartoon,
    body: [0; 4],
    wing: [0; 4],
    belly: [0; 4],
    beak: [0; 4],
    feet: [0; 4],
    cap: None,
    cheek: None,
    bib: None,
    breast: None,
    wing_bar: None,
};

/// The built-in styles; pick one by name (case-insensitive) via `BIRD_STYLE`.
/// The first five are palettes of the cartoon bird; the last five are
/// different designs entirely.
pub const STYLES: [BirdStyle; 10] = [
    // Cartoon, slate blue with a cream belly — the original bird's identity.
    BirdStyle {
        name: "sona",
        body: [84, 104, 148, 255],
        wing: [122, 142, 184, 255],
        belly: [236, 228, 205, 255],
        beak: [232, 156, 46, 255],
        feet: [196, 130, 62, 255],
        ..PLAIN
    },
    // Cartoon chickadee: grey back, black cap and bib, white cheek and belly.
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
        ..PLAIN
    },
    // Cartoon bluebird: vivid blue back and head, rusty chest, white belly.
    BirdStyle {
        name: "laso",
        body: [58, 118, 196, 255],
        wing: [42, 90, 158, 255],
        belly: [246, 243, 236, 255],
        beak: [62, 62, 70, 255],
        feet: [122, 102, 92, 255],
        breast: Some([214, 122, 62, 255]),
        ..PLAIN
    },
    // Cartoon goldfinch: yellow body, black cap, dark wing with a light bar.
    BirdStyle {
        name: "jelo",
        body: [238, 196, 48, 255],
        wing: [56, 58, 64, 255],
        belly: [250, 242, 200, 255],
        beak: [232, 168, 92, 255],
        feet: [182, 142, 102, 255],
        cap: Some([40, 42, 48, 255]),
        wing_bar: Some([248, 246, 238, 255]),
        ..PLAIN
    },
    // Cartoon robin: brown-grey back, red-orange breast, off-white belly.
    BirdStyle {
        name: "loje",
        body: [112, 98, 88, 255],
        wing: [94, 82, 74, 255],
        belly: [240, 235, 225, 255],
        beak: [228, 178, 64, 255],
        feet: [122, 102, 90, 255],
        breast: Some([226, 104, 58, 255]),
        ..PLAIN
    },
    // Round kawaii chick: one big ball, huge glossy eye, blush, stub wing.
    BirdStyle {
        name: "suwi",
        draw: draw_suwi,
        body: [248, 232, 205, 255],
        wing: [232, 205, 168, 255],
        belly: [253, 246, 230, 255],
        beak: [240, 168, 70, 255],
        feet: [222, 152, 84, 255],
        cheek: Some([244, 156, 150, 150]), // translucent blush
        ..PLAIN
    },
    // Retro game sprite: hard 2x2 pixels, no anti-aliasing, black outlines.
    BirdStyle {
        name: "musi",
        draw: draw_musi,
        body: [235, 195, 50, 255],
        wing: [250, 250, 250, 255],
        belly: [250, 240, 210, 255],
        beak: [220, 90, 60, 255],
        feet: [225, 145, 50, 255],
        ..PLAIN
    },
    // Origami: angular paper facets, a triangle wing that folds.
    BirdStyle {
        name: "lipu",
        draw: draw_lipu,
        body: [236, 146, 106, 255],
        wing: [188, 100, 68, 255],
        belly: [250, 244, 234, 255],
        beak: [122, 64, 44, 255],
        feet: [126, 96, 84, 255],
        ..PLAIN
    },
    // Ink doodle: paper-white fills with hand-drawn ink outlines (`wing` is
    // the ink colour).
    BirdStyle {
        name: "sitelen",
        draw: draw_sitelen,
        body: [252, 250, 245, 255],
        wing: [48, 44, 48, 255],
        belly: [252, 250, 245, 255],
        beak: [48, 44, 48, 255],
        feet: [48, 44, 48, 255],
        ..PLAIN
    },
    // Swallow: sleek dark body, long pointed wing, forked streamer tail.
    BirdStyle {
        name: "sewi",
        draw: draw_sewi,
        body: [44, 58, 92, 255],
        wing: [32, 42, 68, 255],
        belly: [246, 246, 242, 255],
        beak: [30, 30, 36, 255],
        feet: [104, 92, 86, 255],
        breast: Some([198, 92, 54, 255]),
        ..PLAIN
    },
];

/// Look up a built-in style by name (case-insensitive), or `None` if unknown.
/// Used by the control panel's live style switch.
pub fn style_by_name(name: &str) -> Option<&'static BirdStyle> {
    STYLES.iter().find(|s| s.name.eq_ignore_ascii_case(name))
}

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
pub struct Pose {
    /// Wing sweep in radians: 0 = trailing along the body, positive = raised.
    pub wing: f32,
    /// Beak opening, 0.0 (closed) ..= 1.0 (wide).
    pub beak: f32,
    /// Vertical settle of the body in pixels; the feet stay planted.
    pub bob: f32,
    /// Legs drawn (standing / perched) or tucked away (flight).
    pub feet: bool,
}

impl Pose {
    pub const FLYING: Pose = Pose { wing: 0.0, beak: 0.0, bob: 0.0, feet: false };
    pub const STANDING: Pose = Pose { wing: -0.12, beak: 0.0, bob: 0.0, feet: true };
}

// --- rasterizer -------------------------------------------------------------

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

/// The annulus between an ellipse grown and shrunk by `w` — an outline stroke
/// of width `2w`.
fn ellipse_ring(cx: f32, cy: f32, rx: f32, ry: f32, rot: f32, w: f32) -> impl Fn(f32, f32) -> bool {
    let outer = ellipse(cx, cy, rx + w, ry + w, rot);
    let inner = ellipse(cx, cy, (rx - w).max(0.1), (ry - w).max(0.1), rot);
    move |x: f32, y: f32| outer(x, y) && !inner(x, y)
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
/// underlays).
fn grow_tri(t: [(f32, f32); 3], d: f32) -> [(f32, f32); 3] {
    let cx = (t[0].0 + t[1].0 + t[2].0) / 3.0;
    let cy = (t[0].1 + t[1].1 + t[2].1) / 3.0;
    t.map(|p| {
        let (vx, vy) = (p.0 - cx, p.1 - cy);
        let l = (vx * vx + vy * vy).sqrt().max(1e-3);
        (p.0 + vx / l * d, p.1 + vy / l * d)
    })
}

/// Rotate `p` about `pivot` by `theta` radians (screen coordinates, y down;
/// positive `theta` swings a leftward point upward).
fn rot_about(p: (f32, f32), pivot: (f32, f32), theta: f32) -> (f32, f32) {
    let (s, c) = theta.sin_cos();
    let (dx, dy) = (p.0 - pivot.0, p.1 - pivot.1);
    (pivot.0 + dx * c - dy * s, pivot.1 + dx * s + dy * c)
}

// --- cartoon ----------------------------------------------------------------

/// The cartoon bird (facing right): plump body, round head, fanned tail,
/// shoulder-pivoting wing. The first five [`STYLES`] are palettes of this.
fn draw_cartoon(st: &BirdStyle, pose: &Pose) -> Frame {
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
    for (tx, ty) in [(3.5, 16.5), (2.5, 21.0), (4.0, 25.5)] {
        let tip = rot_about((tx, ty + b), root, -pose.wing * 0.3);
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

// --- kawaii chick ------------------------------------------------------------

/// A round chick: body and head are one big ball with a huge glossy eye, a
/// blush, and stubs for wing and tail.
fn draw_suwi(st: &BirdStyle, pose: &Pose) -> Frame {
    let mut c = Canvas::new();
    let b = pose.bob;
    let outline = shade(st.body, 0.55);

    // Tail stub behind the ball.
    let tail = [(17.5, 17.5 + b), (17.5, 25.0 + b), (9.5, 21.0 + b)];
    let g = grow_tri(tail, RIM);
    c.fill(outline, &tri(g[0], g[1], g[2]));
    c.fill(shade(st.body, 0.88), &tri(tail[0], tail[1], tail[2]));

    // Tiny feet nubs.
    if pose.feet {
        for lx in [26.0_f32, 33.0] {
            let seg = ((lx, 34.5 + b), (lx + 0.3, 38.0));
            c.fill(shade(st.feet, 0.55), &capsule(seg.0, seg.1, 1.3 + 0.6));
            c.fill(st.feet, &capsule(seg.0, seg.1, 1.3));
        }
    }

    // The ball, then a lighter tummy.
    let ball = circle(30.0, 22.0 + b, 14.5);
    c.fill(outline, &circle(30.0, 22.0 + b, 14.5 + RIM));
    c.fill(st.body, &ball);
    let tummy = ellipse(32.0, 28.0 + b, 9.0, 7.0, 0.1);
    c.fill(st.belly, &|x, y| tummy(x, y) && ball(x, y));

    // Stub wing pivoting at the shoulder.
    let sh = (24.0, 20.5 + b);
    let dir = (-pose.wing.cos(), -pose.wing.sin());
    let wc = (sh.0 + dir.0 * 3.5, sh.1 + dir.1 * 3.5);
    let rot = dir.1.atan2(dir.0);
    c.fill(shade(st.wing, 0.6), &ellipse(wc.0, wc.1, 6.0 + 0.7, 3.4 + 0.7, rot));
    c.fill(st.wing, &ellipse(wc.0, wc.1, 6.0, 3.4, rot));

    // Blush under the eye.
    if let Some(col) = st.cheek {
        c.fill(col, &circle(32.5, 23.5 + b, 2.4));
    }

    // Tiny beak.
    let t = pose.beak;
    let upper = [(43.2, 18.4 + b), (43.2, 20.4 + b), (48.0 - 0.6 * t, 19.2 + b - 2.0 * t)];
    let lower = [(43.2, 19.6 + b), (43.2, 21.4 + b), (47.6 - 0.5 * t, 20.2 + b + 2.2 * t)];
    for (m, col) in [(lower, shade(st.beak, 0.75)), (upper, st.beak)] {
        let g = grow_tri(m, 0.6);
        c.fill(shade(st.beak, 0.5), &tri(g[0], g[1], g[2]));
        c.fill(col, &tri(m[0], m[1], m[2]));
    }

    // One huge glossy eye.
    c.fill(EYE_DARK, &circle(36.8, 16.8 + b, 3.7));
    c.fill(EYE_WHITE, &circle(38.0, 15.4 + b, 1.3));
    c.fill(EYE_WHITE, &circle(35.4, 18.4 + b, 0.7));

    c.frame()
}

// --- retro pixel sprite -------------------------------------------------------

/// An 8-bit-style sprite: drawn on a 28x20 grid with one hard sample per cell
/// (no anti-aliasing), black outlines, then upscaled 2x nearest so every pixel
/// is a chunky 2x2 block. The wing translates instead of rotating.
fn draw_musi(st: &BirdStyle, pose: &Pose) -> Frame {
    const W: i32 = 28;
    const H: i32 = 20;
    const INK: Rgba = [22, 22, 26, 255];
    let mut grid = vec![[0u8; 4]; (W * H) as usize];
    let mut paint = |color: Rgba, inside: &dyn Fn(f32, f32) -> bool| {
        for y in 0..H {
            for x in 0..W {
                if inside(x as f32 + 0.5, y as f32 + 0.5) {
                    grid[(y * W + x) as usize] = color;
                }
            }
        }
    };
    let b = (pose.bob * 0.5).round();

    // Tail wedge, then the one-blob body, all with 1px ink outlines.
    let tail = [(7.0, 7.5 + b), (7.0, 12.5 + b), (2.0, 10.0 + b)];
    let g = grow_tri(tail, 1.0);
    paint(INK, &tri(g[0], g[1], g[2]));
    paint(st.body, &tri(tail[0], tail[1], tail[2]));
    let body = ellipse(13.0, 10.0 + b, 8.0, 6.5, 0.0);
    paint(INK, &ellipse(13.0, 10.0 + b, 9.0, 7.5, 0.0));
    paint(st.body, &body);
    let belly = ellipse(15.0, 13.5 + b, 6.0, 4.0, 0.0);
    paint(st.belly, &|x, y| belly(x, y) && body(x, y));

    // Wing: an outlined oval that hops up and down with the flap.
    let wy = 11.0 + b - (pose.wing * 4.0).round();
    paint(INK, &ellipse(9.0, wy, 5.5, 3.8, 0.0));
    paint(st.wing, &ellipse(9.0, wy, 4.5, 2.8, 0.0));

    // Big white eye with an offset pupil.
    paint(INK, &circle(17.8, 6.5 + b, 3.4));
    paint(EYE_WHITE, &circle(17.8, 6.5 + b, 2.6));
    paint(INK, &circle(18.8, 6.6 + b, 1.1));

    // Lips-style beak: two stacked ovals that part when chirping.
    let t = pose.beak;
    paint(INK, &ellipse(23.0, 11.0 + b - 1.5 * t, 4.6, 2.5, 0.0));
    paint(INK, &ellipse(22.5, 12.5 + b + 1.5 * t, 4.0, 2.1, 0.0));
    paint(st.beak, &ellipse(23.0, 11.0 + b - 1.5 * t, 3.8, 1.7, 0.0));
    paint(shade(st.beak, 0.8), &ellipse(22.5, 12.5 + b + 1.5 * t, 3.2, 1.3, 0.0));

    // Blocky feet.
    if pose.feet {
        for cx in [10.5_f32, 15.5] {
            paint(st.feet, &move |x: f32, y: f32| {
                (x - cx).abs() <= 1.4 && y >= 16.8 + b && y <= 18.8
            });
        }
    }

    // Upscale x2 nearest into the shared canvas size.
    let mut pixels = vec![0u8; (PW * PH * 4) as usize];
    for y in 0..PH as i32 {
        for x in 0..PW as i32 {
            let c = grid[((y / 2) * W + x / 2) as usize];
            let o = ((y * PW as i32 + x) * 4) as usize;
            pixels[o..o + 4].copy_from_slice(&c);
        }
    }
    Frame { w: PW, h: PH, pixels }
}

// --- origami -----------------------------------------------------------------

/// A folded-paper bird: everything is flat triangles, with light and shadow
/// facets suggesting the folds and a big triangle wing that creases upward.
fn draw_lipu(st: &BirdStyle, pose: &Pose) -> Frame {
    let mut c = Canvas::new();
    let b = pose.bob;
    let edge = shade(st.body, 0.5);
    let facet = |col: Rgba, t: [(f32, f32); 3], c: &mut Canvas| {
        let g = grow_tri(t, 0.5);
        c.fill(edge, &tri(g[0], g[1], g[2]));
        c.fill(col, &tri(t[0], t[1], t[2]));
    };

    // Tail spike.
    facet(shade(st.wing, 0.9), [(18.0, 20.0 + b), (18.0, 27.0 + b), (4.0, 23.5 + b)], &mut c);

    // Stick legs with angular feet.
    if pose.feet {
        for lx in [23.5_f32, 29.0] {
            let knee = (lx - 0.8, 35.6);
            for (p, q) in [((lx, 31.0 + b), knee), (knee, (knee.0 + 3.0, 37.4))] {
                c.fill(st.feet, &capsule(p, q, 0.9));
            }
        }
    }

    // Body: a kite split into a light top facet and a shadow bottom facet.
    facet(st.body, [(11.0, 29.0 + b), (27.0, 10.0 + b), (45.0, 21.5 + b)], &mut c);
    facet(st.wing, [(11.0, 29.0 + b), (45.0, 21.5 + b), (33.0, 33.0 + b)], &mut c);

    // Head: a diamond, also split light/shadow.
    facet(st.body, [(31.0, 12.0 + b), (40.0, 4.5 + b), (47.5, 12.5 + b)], &mut c);
    facet(st.wing, [(31.0, 12.0 + b), (47.5, 12.5 + b), (38.0, 18.5 + b)], &mut c);

    // Wing: one big triangle creasing about the shoulder.
    let pivot = (29.0, 15.0 + b);
    let theta = pose.wing + 0.12; // folded pose is the rest position
    let w: Vec<(f32, f32)> = [(1.5, -1.5), (3.0, 6.0), (-20.0, 9.0)]
        .iter()
        .map(|v| rot_about((pivot.0 + v.0, pivot.1 + v.1), pivot, theta))
        .collect();
    facet(shade(st.body, 1.12), [w[0], w[1], w[2]], &mut c);

    // Beak: two slivers that part when chirping (they meet when closed).
    let t = pose.beak;
    facet(st.beak, [(46.0, 10.9 + b), (47.4, 13.2 + b), (54.2 - 1.2 * t, 12.0 + b - 2.8 * t)], &mut c);
    facet(
        shade(st.beak, 0.8),
        [(46.2, 13.0 + b), (47.4, 15.0 + b), (53.4 - 1.0 * t, 12.8 + b + 3.0 * t)],
        &mut c,
    );

    // A dot eye on the light head facet.
    c.fill(EYE_DARK, &circle(40.5, 11.0 + b, 1.3));

    c.frame()
}

// --- ink doodle ----------------------------------------------------------------

/// A margin doodle: paper-white fills traced with ink outlines, scribbled tail
/// and feather lines, stick legs, dot eye. `st.wing` is the ink colour.
fn draw_sitelen(st: &BirdStyle, pose: &Pose) -> Frame {
    let mut c = Canvas::new();
    let b = pose.bob;
    let ink = st.wing;
    const LINE: f32 = 0.6; // stroke half-width

    // Tail: three quick line strokes fanning from the rump.
    let root = (16.0, 21.5 + b);
    for (tx, ty) in [(4.5, 16.5), (3.5, 21.5), (4.5, 26.5)] {
        let tip = rot_about((tx, ty + b), root, -pose.wing * 0.3);
        c.fill(ink, &capsule(root, tip, 0.65));
    }

    // Stick legs with two toes each.
    if pose.feet {
        for lx in [23.0_f32, 28.5] {
            let ankle = (lx - 0.4, 37.4);
            for (p, q) in [
                ((lx, 33.0 + b), ankle),
                (ankle, (ankle.0 + 2.6, 38.1)),
                (ankle, (ankle.0 - 2.2, 38.0)),
            ] {
                c.fill(ink, &capsule(p, q, 0.7));
            }
        }
    }

    // Body and head: paper fill, then ink outlines that stop where they pass
    // inside the other circle so the union reads as one stroke.
    let (bc, br) = ((25.5_f32, 24.0 + b), 11.0_f32);
    let (hc, hr) = ((37.5_f32, 14.0 + b), 7.4_f32);
    let body = circle(bc.0, bc.1, br);
    let head = circle(hc.0, hc.1, hr);
    c.fill(st.body, &|x, y| body(x, y) || head(x, y));
    let body_ring = ellipse_ring(bc.0, bc.1, br, br, 0.0, LINE);
    let head_inner = circle(hc.0, hc.1, hr - LINE);
    c.fill(ink, &|x, y| body_ring(x, y) && !head_inner(x, y));
    let head_ring = ellipse_ring(hc.0, hc.1, hr, hr, 0.0, LINE);
    let body_inner = circle(bc.0, bc.1, br - LINE);
    c.fill(ink, &|x, y| head_ring(x, y) && !body_inner(x, y));

    // Wing: an outlined oval pivoting at the shoulder, with two feather lines.
    let sh = (26.0, 19.0 + b);
    let dir = (-pose.wing.cos(), -pose.wing.sin());
    let perp = (dir.1, -dir.0);
    let wc = (sh.0 + dir.0 * 5.5, sh.1 + dir.1 * 5.5);
    let rot = dir.1.atan2(dir.0);
    let wing = ellipse(wc.0, wc.1, 9.0, 4.2, rot);
    c.fill(st.body, &wing);
    c.fill(ink, &ellipse_ring(wc.0, wc.1, 9.0, 4.2, rot, LINE));
    for out in [0.4_f32, 2.0] {
        let p = (wc.0 + dir.0 * 2.0 + perp.0 * out, wc.1 + dir.1 * 2.0 + perp.1 * out);
        let q = (p.0 + dir.0 * 5.5, p.1 + dir.1 * 5.5);
        let line = capsule(p, q, 0.55);
        let inner = ellipse(wc.0, wc.1, 9.0 - LINE, 4.2 - LINE, rot);
        c.fill(ink, &|x, y| line(x, y) && inner(x, y));
    }

    // Solid ink beak, splitting open to chirp.
    let t = pose.beak;
    let upper = [(44.0, 12.4 + b), (44.0, 14.6 + b), (50.4 - 0.8 * t, 13.6 + b - 2.6 * t)];
    let lower = [(44.0, 13.8 + b), (44.0, 15.8 + b), (49.8 - 0.7 * t, 14.4 + b + 2.8 * t)];
    c.fill(ink, &tri(upper[0], upper[1], upper[2]));
    c.fill(ink, &tri(lower[0], lower[1], lower[2]));

    // Dot eye.
    c.fill(ink, &circle(40.0, 12.6 + b, 1.6));

    c.frame()
}

// --- swallow -------------------------------------------------------------------

/// A sleek swallow: slim dark body, rust throat, white underside, one long
/// pointed wing, and a forked streamer tail.
fn draw_sewi(st: &BirdStyle, pose: &Pose) -> Frame {
    let mut c = Canvas::new();
    let b = pose.bob;
    let outline = shade(st.body, 0.45);

    // Forked tail: two long thin streamers.
    let root = (13.0, 21.5 + b);
    for (tx, ty) in [(1.5, 14.5), (1.5, 24.5)] {
        let tip = rot_about((tx, ty + b), root, -pose.wing * 0.3);
        c.fill(outline, &capsule(root, tip, 1.6 + 0.7));
        c.fill(shade(st.body, 0.8), &capsule(root, tip, 1.6));
    }

    // Short legs.
    if pose.feet {
        for lx in [24.0_f32, 29.0] {
            let ankle = (lx + 0.4, 35.8);
            let segs = [
                ((lx, 30.5 + b), ankle, 1.0),
                (ankle, (ankle.0 + 2.4, 36.6), 0.8),
                (ankle, (ankle.0 - 2.0, 36.4), 0.8),
            ];
            for (p, q, r) in segs {
                c.fill(shade(st.feet, 0.5), &capsule(p, q, r + 0.5));
            }
            for (p, q, r) in segs {
                c.fill(st.feet, &capsule(p, q, r));
            }
        }
    }

    // Slim teardrop torso.
    let body = ellipse(25.0, 23.0 + b, 14.0, 7.5, -0.20);
    let neck = ellipse(32.0, 16.5 + b, 7.0, 6.0, 0.0);
    let head = circle(38.5, 13.0 + b, 6.6);
    let torso = |x: f32, y: f32| body(x, y) || neck(x, y) || head(x, y);
    c.fill(outline, &ellipse(25.0, 23.0 + b, 14.0 + RIM, 7.5 + RIM, -0.20));
    c.fill(outline, &circle(38.5, 13.0 + b, 6.6 + RIM));
    c.fill(st.body, &torso);

    // Rust throat, then the white underside over its lower edge.
    if let Some(col) = st.breast {
        let patch = ellipse(38.0, 18.5 + b, 5.0, 4.5, 0.3);
        c.fill(col, &|x, y| patch(x, y) && torso(x, y));
    }
    let belly = ellipse(27.0, 28.5 + b, 10.5, 5.5, 0.15);
    c.fill(st.belly, &|x, y| belly(x, y) && torso(x, y));

    // One long pointed wing sweeping from the shoulder, with a lighter covert
    // patch at its base.
    let pivot = (28.0, 17.0 + b);
    let theta = pose.wing + 0.12; // folded along the body at rest
    let wing_pts =
        [(1.5, -2.5), (2.5, 4.0), (-23.0, 2.0)].map(|v| rot_about((pivot.0 + v.0, pivot.1 + v.1), pivot, theta));
    let g = grow_tri(wing_pts, 0.7);
    c.fill(outline, &tri(g[0], g[1], g[2]));
    c.fill(st.wing, &tri(wing_pts[0], wing_pts[1], wing_pts[2]));
    let covert =
        [(1.0, -2.0), (2.0, 3.0), (-9.0, 2.5)].map(|v| rot_about((pivot.0 + v.0, pivot.1 + v.1), pivot, theta));
    c.fill(shade(st.body, 1.45), &tri(covert[0], covert[1], covert[2]));

    // Short dark beak.
    let t = pose.beak;
    let upper = [(44.2, 12.3 + b), (44.2, 14.6 + b), (49.0 - 0.6 * t, 13.4 + b - 1.8 * t)];
    let lower = [(44.2, 13.8 + b), (44.2, 15.6 + b), (48.4 - 0.5 * t, 14.6 + b + 2.4 * t)];
    c.fill(st.beak, &tri(upper[0], upper[1], upper[2]));
    c.fill(shade(st.beak, 0.8), &tri(lower[0], lower[1], lower[2]));

    // Small natural eye with a light ring.
    c.fill(EYE_WHITE, &circle(40.8, 11.8 + b, 2.3));
    c.fill(EYE_DARK, &circle(40.8, 11.8 + b, 1.8));
    c.fill(EYE_WHITE, &circle(41.4, 11.2 + b, 0.55));

    c.frame()
}

// --- style sheet -----------------------------------------------------------------

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
            let f = st.frame(pose);
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
