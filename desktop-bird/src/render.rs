//! Software framebuffer helpers: rectangle math, sprite blitting, and speech
//! bubble rendering.
//!
//! The bird lives in a transparent, output-sized `Argb8888` buffer. `wl_shm`'s
//! `Argb8888` is little-endian and **premultiplied alpha**, so in memory each
//! pixel is `[B, G, R, A]` with the colour channels already scaled by alpha.

use crate::font;
use crate::sprite::Frame;
use bird_protocol::BubbleTuning;

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
// Speech bubble (sitelen pona, drawn with the bundled nasin-nanpa font)
// ---------------------------------------------------------------------------

/// Draw a speech bubble above (or below, if too close to the top) the bird.
///
/// `text` is UTF-8 sitelen pona (UCSUR); `bird_{x,y}` is the top-left of the
/// sprite and `bird_{w,h}` its size. The look (text size, padding, tail, corner
/// radius, ink) comes from `t` — `BubbleTuning::default()` reproduces the
/// original constants. Returns the bounding [`Rect`] of the whole bubble (box +
/// tail + shadow) so the caller can union it into the damage region.
pub fn draw_bubble(
    canvas: &mut [u8],
    buf_w: u32,
    buf_h: u32,
    text: &str,
    bird_x: i32,
    bird_y: i32,
    bird_w: i32,
    bird_h: i32,
    t: &BubbleTuning,
) -> Rect {
    // Bind the tuned values once; the body below reads them like the old consts.
    let text_px = t.text_px.max(1.0);
    let pad = t.pad.max(0);
    let tail = t.tail.max(0);
    let radius = t.radius.max(0);
    let ink = t.ink.min(255);

    let line = font::layout(text, text_px);

    // Measure text, capped so the bubble never exceeds half the screen width.
    let max_text_w = (buf_w as i32 / 2).max(80);
    let tw = line.width.min(max_text_w).max(1);
    let th = line.height.max(1);

    let bw = tw + pad * 2 + 2; // +2 for the 1px border on each side
    let bh = th + pad * 2 + 2;

    // Horizontal: centre over the bird, clamped within the buffer.
    let cx = bird_x + bird_w / 2;
    let bx = (cx - bw / 2).clamp(0, (buf_w as i32 - bw).max(0));

    // Vertical: prefer above; fall back to below.
    let (by, tail_below) = {
        let above = bird_y - bh - tail;
        if above >= 0 {
            (above, true)
        } else {
            (bird_y + bird_h + tail, false)
        }
    };

    // Soft drop shadow (rounded, translucent), offset down-right, behind the box.
    rounded_rect(canvas, buf_w, buf_h, bx + 2, by + 2, bw, bh, Some((0, 0, 0, 0x40)), None, radius);
    // White body + dark rounded border.
    rounded_rect(
        canvas, buf_w, buf_h, bx, by, bw, bh,
        Some((0xFF, 0xFF, 0xFF, 0xFF)), Some((0x44, 0x44, 0x44, 0xFF)), radius,
    );

    // Triangular tail. The tip points toward the bird's centre line, kept clear
    // of the rounded corners.
    let tail_cx = cx.clamp(bx + tail + radius, bx + bw - tail - radius);
    for i in 0..tail {
        let half = tail - 1 - i; // widest near the bubble, narrowest at the tip
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

    // Composite the rasterized sitelen pona over the opaque white interior.
    let text_x = bx + 1 + pad;
    let text_y = by + 1 + pad;
    let x_min = bx + 1;
    let x_max = bx + bw - 2;
    for g in &line.glyphs {
        for gy in 0..g.h {
            let y = text_y + g.dy + gy as i32;
            for gx in 0..g.w {
                let c = g.bitmap[gy * g.w + gx] as u32;
                if c == 0 {
                    continue;
                }
                let x = text_x + g.dx + gx as i32;
                if x < x_min || x > x_max {
                    continue; // clip to the interior when the text was width-capped
                }
                // Ink over opaque white: out = white*(255-c)/255 + ink*c/255.
                let out = ((255 * (255 - c) + ink * c) / 255) as u8;
                put_pixel(canvas, buf_w, buf_h, x, y, out, out, out, 0xFF);
            }
        }
    }

    // Bounding rect covering box, tail, and the +2 shadow.
    let top = if tail_below { by } else { by - tail };
    Rect { x: bx, y: top, w: bw + 2, h: bh + tail + 2 }
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

/// Is local pixel `(dx, dy)` inside a `w x h` rounded rectangle of corner radius
/// `rad`? Corners are quarter-circles of radius `rad`; edges/interior are flat.
fn rr_inside(dx: i32, dy: i32, w: i32, h: i32, rad: i32) -> bool {
    if dx < 0 || dy < 0 || dx >= w || dy >= h {
        return false;
    }
    if rad <= 0 {
        return true;
    }
    // Clamp to the nearest corner-circle centre; on the flat edges this collapses
    // to the same row/column so the test is always satisfied there.
    let ix = dx.clamp(rad, (w - 1 - rad).max(rad));
    let iy = dy.clamp(rad, (h - 1 - rad).max(rad));
    let ddx = dx - ix;
    let ddy = dy - iy;
    ddx * ddx + ddy * ddy <= rad * rad
}

/// Fill and/or outline a rounded rectangle at `(x, y)` of size `w x h`.
fn rounded_rect(
    canvas: &mut [u8],
    buf_w: u32,
    buf_h: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    fill: Option<(u8, u8, u8, u8)>,
    border: Option<(u8, u8, u8, u8)>,
    rad: i32,
) {
    for dy in 0..h {
        for dx in 0..w {
            if !rr_inside(dx, dy, w, h, rad) {
                continue;
            }
            // Border pixels are inside cells with at least one outside 4-neighbour.
            if let Some((r, g, b, a)) = border {
                let edge = !rr_inside(dx - 1, dy, w, h, rad)
                    || !rr_inside(dx + 1, dy, w, h, rad)
                    || !rr_inside(dx, dy - 1, w, h, rad)
                    || !rr_inside(dx, dy + 1, w, h, rad);
                if edge {
                    put_pixel(canvas, buf_w, buf_h, x + dx, y + dy, r, g, b, a);
                    continue;
                }
            }
            if let Some((r, g, b, a)) = fill {
                put_pixel(canvas, buf_w, buf_h, x + dx, y + dy, r, g, b, a);
            }
        }
    }
}
