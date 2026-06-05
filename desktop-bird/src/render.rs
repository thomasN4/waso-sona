//! Software framebuffer helpers: rectangle math, sprite blitting, and speech
//! bubble rendering.
//!
//! The bird lives in a transparent, output-sized `Argb8888` buffer. `wl_shm`'s
//! `Argb8888` is little-endian and **premultiplied alpha**, so in memory each
//! pixel is `[B, G, R, A]` with the colour channels already scaled by alpha.

use crate::sprite::Frame;

/// An axis-aligned rectangle in buffer pixels.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    /// Smallest rectangle covering both `self` and `other`.
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.w).max(other.x + other.w);
        let bottom = (self.y + self.h).max(other.y + other.h);
        Rect { x, y, w: right - x, h: bottom - y }
    }

    /// Clip the rectangle to the `[0, w) x [0, h)` buffer, so it is always a
    /// valid `damage_buffer` argument. May produce a zero-size rect.
    pub fn clamp(self, buf_w: u32, buf_h: u32) -> Rect {
        let bw = buf_w as i32;
        let bh = buf_h as i32;
        let x = self.x.clamp(0, bw);
        let y = self.y.clamp(0, bh);
        let right = (self.x + self.w).clamp(0, bw);
        let bottom = (self.y + self.h).clamp(0, bh);
        Rect { x, y, w: (right - x).max(0), h: (bottom - y).max(0) }
    }
}

/// Blit one sprite frame into `canvas` at top-left `(ox, oy)`.
///
/// `canvas` is `buf_w * buf_h` pixels of `Argb8888` (premultiplied BGRA bytes).
/// Source pixels are straight (non-premultiplied) `RGBA8`. Fully transparent
/// source pixels are skipped; everything else is premultiplied on the way in.
/// When `flip` is set the frame is mirrored horizontally (the placeholder bird
/// faces right by default, so it flips when travelling left).
pub fn blit(canvas: &mut [u8], buf_w: u32, buf_h: u32, frame: &Frame, ox: i32, oy: i32, flip: bool) {
    let bw = buf_w as i32;
    let bh = buf_h as i32;
    for sy in 0..frame.h as i32 {
        let dy = oy + sy;
        if dy < 0 || dy >= bh {
            continue;
        }
        for sx in 0..frame.w as i32 {
            let src_x = if flip { frame.w as i32 - 1 - sx } else { sx };
            let si = ((sy * frame.w as i32 + src_x) * 4) as usize;
            let a = frame.pixels[si + 3];
            if a == 0 {
                continue;
            }
            let dx = ox + sx;
            if dx < 0 || dx >= bw {
                continue;
            }
            let r = frame.pixels[si];
            let g = frame.pixels[si + 1];
            let b = frame.pixels[si + 2];
            let di = ((dy * bw + dx) * 4) as usize;
            // Premultiply, write little-endian Argb8888 => [B, G, R, A].
            canvas[di] = premul(b, a);
            canvas[di + 1] = premul(g, a);
            canvas[di + 2] = premul(r, a);
            canvas[di + 3] = a;
        }
    }
}

#[inline]
fn premul(channel: u8, alpha: u8) -> u8 {
    ((channel as u16 * alpha as u16) / 255) as u8
}

// ---------------------------------------------------------------------------
// Speech bubble
// ---------------------------------------------------------------------------

/// Glyph width and height for the embedded 5×7 pixel font.
const GW: i32 = 5;
const GH: i32 = 7;
/// One extra pixel of horizontal gap between characters.
const CSTRIDE: i32 = GW + 1;

/// Draw a speech bubble above (or below, if too close to the top) the bird.
///
/// `bird_{x,y}` is the top-left of the sprite; `bird_{w,h}` its size.
/// Returns the bounding [`Rect`] of the whole bubble (box + tail) so the
/// caller can union it into the damage region.
pub fn draw_bubble(
    canvas: &mut [u8],
    buf_w: u32,
    buf_h: u32,
    text: &str,
    bird_x: i32,
    bird_y: i32,
    bird_w: i32,
    bird_h: i32,
) -> Rect {
    const PAD: i32 = 4; // inner padding around text
    const TAIL: i32 = 5; // height of the triangular tail

    // Measure text, capped so the bubble never exceeds half the screen width.
    let max_text_w = (buf_w as i32 / 2).max(80);
    let raw_tw = text_pixel_width(text);
    let tw = raw_tw.min(max_text_w);

    let bw = tw + PAD * 2 + 2; // +2 for 1px border on each side
    let bh = GH + PAD * 2 + 2;

    // Horizontal: centre over the bird, clamped within the buffer.
    let cx = bird_x + bird_w / 2;
    let bx = (cx - bw / 2).clamp(0, (buf_w as i32 - bw).max(0));

    // Vertical: prefer above; fall back to below.
    let (by, tail_below) = {
        let above = bird_y - bh - TAIL;
        if above >= 0 {
            (above, true)
        } else {
            (bird_y + bird_h + TAIL, false)
        }
    };

    // Background (white, opaque).
    fill_rect(canvas, buf_w, buf_h, bx, by, bw, bh, 0xFF, 0xFF, 0xFF, 0xFF);
    // Border (dark).
    outline_rect(canvas, buf_w, buf_h, bx, by, bw, bh, 0x44, 0x44, 0x44);

    // Triangular tail.  The tip points toward the bird's centre line.
    let tail_cx = cx.clamp(bx + TAIL, bx + bw - TAIL);
    for i in 0..TAIL {
        let half = TAIL - 1 - i; // widest near the bubble, narrowest at tip
        let ty = if tail_below { by + bh + i } else { by - 1 - i };
        for tx in (tail_cx - half)..=(tail_cx + half) {
            put_pixel(canvas, buf_w, buf_h, tx, ty, 0xFF, 0xFF, 0xFF, 0xFF);
        }
        // Side border strokes (skip when half == 0, that's just the tip pixel).
        if half > 0 {
            put_pixel(canvas, buf_w, buf_h, tail_cx - half, ty, 0x44, 0x44, 0x44, 0xFF);
            put_pixel(canvas, buf_w, buf_h, tail_cx + half, ty, 0x44, 0x44, 0x44, 0xFF);
        }
    }

    // Text, clipped to the interior.
    let text_x = bx + 1 + PAD;
    let text_y = by + 1 + PAD;
    let right_limit = bx + bw - 1 - PAD;
    let mut dx = text_x;
    for ch in text.chars() {
        if dx + GW > right_limit {
            break;
        }
        draw_glyph(canvas, buf_w, buf_h, ch, dx, text_y, 0x22, 0x22, 0x22);
        dx += CSTRIDE;
    }

    // Bounding rect (bubble box + tail).
    if tail_below {
        Rect { x: bx, y: by, w: bw, h: bh + TAIL }
    } else {
        Rect { x: bx, y: by - TAIL, w: bw, h: bh + TAIL }
    }
}

// --- pixel helpers ----------------------------------------------------------

#[inline]
fn put_pixel(canvas: &mut [u8], buf_w: u32, buf_h: u32, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8) {
    if x < 0 || y < 0 || x >= buf_w as i32 || y >= buf_h as i32 {
        return;
    }
    let i = ((y * buf_w as i32 + x) * 4) as usize;
    canvas[i] = premul(b, a);
    canvas[i + 1] = premul(g, a);
    canvas[i + 2] = premul(r, a);
    canvas[i + 3] = a;
}

fn fill_rect(canvas: &mut [u8], buf_w: u32, buf_h: u32, x: i32, y: i32, w: i32, h: i32, r: u8, g: u8, b: u8, a: u8) {
    for dy in 0..h {
        for dx in 0..w {
            put_pixel(canvas, buf_w, buf_h, x + dx, y + dy, r, g, b, a);
        }
    }
}

fn outline_rect(canvas: &mut [u8], buf_w: u32, buf_h: u32, x: i32, y: i32, w: i32, h: i32, r: u8, g: u8, b: u8) {
    for dx in 0..w {
        put_pixel(canvas, buf_w, buf_h, x + dx, y, r, g, b, 0xFF);
        put_pixel(canvas, buf_w, buf_h, x + dx, y + h - 1, r, g, b, 0xFF);
    }
    for dy in 1..(h - 1) {
        put_pixel(canvas, buf_w, buf_h, x, y + dy, r, g, b, 0xFF);
        put_pixel(canvas, buf_w, buf_h, x + w - 1, y + dy, r, g, b, 0xFF);
    }
}

fn draw_glyph(canvas: &mut [u8], buf_w: u32, buf_h: u32, ch: char, ox: i32, oy: i32, r: u8, g: u8, b: u8) {
    if !ch.is_ascii() {
        return;
    }
    let code = ch as usize;
    if code < 32 || code > 126 {
        return;
    }
    let glyph = FONT[code - 32];
    for row in 0..GH as usize {
        let bits = glyph[row];
        for col in 0..GW as usize {
            // MSB (bit 4) = leftmost pixel.
            if bits & (0x10u8 >> col) != 0 {
                put_pixel(canvas, buf_w, buf_h, ox + col as i32, oy + row as i32, r, g, b, 0xFF);
            }
        }
    }
}

fn text_pixel_width(s: &str) -> i32 {
    let n = s.chars().filter(|c| c.is_ascii()).count() as i32;
    if n == 0 { 0 } else { n * CSTRIDE - 1 }
}

// --- embedded 5×7 pixel font (printable ASCII 32–126) ----------------------
//
// Each entry is one character: 7 rows, each row is a u8 where the lower 5
// bits encode pixels left-to-right (bit 4 = leftmost, bit 0 = rightmost).

#[rustfmt::skip]
const FONT: [[u8; 7]; 95] = [
    [0, 0, 0, 0, 0, 0, 0],          // 32  space
    [4, 4, 4, 4, 4, 0, 4],           // 33  !
    [10, 10, 0, 0, 0, 0, 0],         // 34  "
    [10, 31, 10, 10, 31, 10, 0],     // 35  #
    [4, 15, 20, 14, 5, 30, 4],       // 36  $
    [25, 25, 2, 4, 8, 19, 19],       // 37  %
    [12, 18, 20, 8, 21, 18, 13],     // 38  &
    [4, 4, 0, 0, 0, 0, 0],           // 39  '
    [2, 4, 8, 8, 8, 4, 2],           // 40  (
    [8, 4, 2, 2, 2, 4, 8],           // 41  )
    [0, 4, 21, 14, 21, 4, 0],        // 42  *
    [0, 4, 4, 31, 4, 4, 0],          // 43  +
    [0, 0, 0, 0, 12, 4, 8],          // 44  ,
    [0, 0, 0, 31, 0, 0, 0],          // 45  -
    [0, 0, 0, 0, 0, 12, 12],         // 46  .
    [1, 2, 4, 8, 16, 0, 0],          // 47  /
    [14, 17, 19, 21, 25, 17, 14],    // 48  0
    [4, 12, 4, 4, 4, 4, 14],         // 49  1
    [14, 17, 1, 6, 8, 16, 31],       // 50  2
    [15, 1, 1, 7, 1, 1, 15],         // 51  3
    [2, 6, 10, 18, 31, 2, 2],        // 52  4
    [31, 16, 30, 1, 1, 17, 14],      // 53  5
    [6, 8, 16, 30, 17, 17, 14],      // 54  6
    [31, 1, 2, 4, 8, 8, 8],          // 55  7
    [14, 17, 17, 14, 17, 17, 14],    // 56  8
    [14, 17, 17, 15, 1, 2, 12],      // 57  9
    [0, 12, 12, 0, 12, 12, 0],       // 58  :
    [0, 12, 12, 0, 12, 4, 8],        // 59  ;
    [2, 4, 8, 16, 8, 4, 2],          // 60  <
    [0, 0, 31, 0, 31, 0, 0],         // 61  =
    [16, 8, 4, 2, 4, 8, 16],         // 62  >
    [14, 17, 1, 6, 4, 0, 4],         // 63  ?
    [14, 17, 23, 21, 22, 16, 14],    // 64  @
    [14, 17, 17, 31, 17, 17, 17],    // 65  A
    [30, 17, 17, 30, 17, 17, 30],    // 66  B
    [14, 17, 16, 16, 16, 17, 14],    // 67  C
    [28, 18, 17, 17, 17, 18, 28],    // 68  D
    [31, 16, 16, 30, 16, 16, 31],    // 69  E
    [31, 16, 16, 30, 16, 16, 16],    // 70  F
    [14, 17, 16, 23, 17, 17, 14],    // 71  G
    [17, 17, 17, 31, 17, 17, 17],    // 72  H
    [14, 4, 4, 4, 4, 4, 14],         // 73  I
    [7, 2, 2, 2, 2, 18, 12],         // 74  J
    [17, 18, 20, 24, 20, 18, 17],    // 75  K
    [16, 16, 16, 16, 16, 16, 31],    // 76  L
    [17, 27, 21, 21, 17, 17, 17],    // 77  M
    [17, 25, 21, 19, 17, 17, 17],    // 78  N
    [14, 17, 17, 17, 17, 17, 14],    // 79  O
    [30, 17, 17, 30, 16, 16, 16],    // 80  P
    [14, 17, 17, 17, 21, 18, 13],    // 81  Q
    [30, 17, 17, 30, 20, 18, 17],    // 82  R
    [14, 17, 16, 14, 1, 17, 14],     // 83  S
    [31, 4, 4, 4, 4, 4, 4],          // 84  T
    [17, 17, 17, 17, 17, 17, 14],    // 85  U
    [17, 17, 17, 17, 17, 10, 4],     // 86  V
    [17, 17, 17, 21, 21, 27, 17],    // 87  W
    [17, 17, 10, 4, 10, 17, 17],     // 88  X
    [17, 17, 10, 4, 4, 4, 4],        // 89  Y
    [31, 1, 2, 4, 8, 16, 31],        // 90  Z
    [14, 8, 8, 8, 8, 8, 14],         // 91  [
    [16, 8, 4, 2, 1, 0, 0],          // 92  backslash
    [14, 2, 2, 2, 2, 2, 14],         // 93  ]
    [4, 10, 17, 0, 0, 0, 0],         // 94  ^
    [0, 0, 0, 0, 0, 0, 31],          // 95  _
    [8, 4, 0, 0, 0, 0, 0],           // 96  `
    [0, 0, 14, 1, 15, 17, 15],       // 97  a
    [16, 16, 30, 17, 17, 17, 30],    // 98  b
    [0, 0, 14, 16, 16, 17, 14],      // 99  c
    [1, 1, 15, 17, 17, 17, 15],      // 100 d
    [0, 0, 14, 17, 31, 16, 14],      // 101 e
    [6, 8, 30, 8, 8, 8, 8],          // 102 f
    [0, 0, 15, 17, 15, 1, 14],       // 103 g
    [16, 16, 30, 17, 17, 17, 17],    // 104 h
    [4, 0, 4, 4, 4, 4, 14],          // 105 i
    [2, 0, 2, 2, 2, 2, 12],          // 106 j
    [16, 16, 18, 20, 24, 20, 18],    // 107 k
    [12, 4, 4, 4, 4, 4, 14],         // 108 l
    [0, 0, 26, 21, 21, 17, 17],      // 109 m
    [0, 0, 30, 17, 17, 17, 17],      // 110 n
    [0, 0, 14, 17, 17, 17, 14],      // 111 o
    [0, 0, 30, 17, 17, 30, 16],      // 112 p
    [0, 0, 15, 17, 17, 15, 1],       // 113 q
    [0, 0, 22, 24, 16, 16, 16],      // 114 r
    [0, 0, 14, 16, 14, 1, 14],       // 115 s
    [4, 4, 30, 4, 4, 4, 6],          // 116 t
    [0, 0, 17, 17, 17, 17, 14],      // 117 u
    [0, 0, 17, 17, 17, 10, 4],       // 118 v
    [0, 0, 17, 21, 21, 21, 10],      // 119 w
    [0, 0, 17, 10, 4, 10, 17],       // 120 x
    [0, 0, 17, 17, 15, 1, 14],       // 121 y
    [0, 0, 31, 2, 4, 8, 31],         // 122 z
    [2, 4, 4, 8, 4, 4, 2],           // 123 {
    [4, 4, 4, 4, 4, 4, 4],           // 124 |
    [8, 4, 4, 2, 4, 4, 8],           // 125 }
    [0, 8, 21, 2, 0, 0, 0],          // 126 ~
];
