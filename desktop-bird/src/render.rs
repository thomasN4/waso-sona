//! Software framebuffer helpers: rectangle math and sprite blitting.
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
