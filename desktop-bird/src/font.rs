//! sitelen pona text rendering for speech bubbles.
//!
//! The bird is a thin display client: it shows sitelen pona (UCSUR) text that
//! the Python side hands it (see `sitelen/translate.py` — `latin_to_ucsur`).
//! Here we rasterize that UCSUR text with the bundled **nasin-nanpa** font
//! (© jan Itan, MIT — `assets/fonts/`) so the bubble shows real glyphs rather
//! than Latin letters.
//!
//! No OpenType shaping is done (no GSUB), so cartouches / long-glyph ligatures
//! won't combine — but each Toki Pona word is a single UCSUR codepoint (one
//! outline), so plain word text rasterizes correctly per-glyph. That's all the
//! bird needs; names / long glyphs are future work.

use std::sync::OnceLock;

use fontdue::Font;

/// The UCSUR-only nasin-nanpa OpenType font, embedded at build time.
const NASIN_NANPA: &[u8] = include_bytes!("../assets/fonts/nasin-nanpa-4.0.2-UCSUR.otf");

/// The parsed font, built once on first use.
pub fn font() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(NASIN_NANPA, fontdue::FontSettings::default())
            .expect("bundled nasin-nanpa OTF failed to parse")
    })
}

/// One rasterized glyph, placed relative to the line's top-left corner.
pub struct Glyph {
    /// 8-bit coverage (alpha), `w * h`, row-major.
    pub bitmap: Vec<u8>,
    pub w: usize,
    pub h: usize,
    /// Top-left offset of `bitmap` within the line box.
    pub dx: i32,
    pub dy: i32,
}

/// A laid-out, rasterized line of UCSUR text, ready to composite into a bubble.
/// Glyph offsets are already baked relative to the line-box top-left.
pub struct RasterLine {
    pub glyphs: Vec<Glyph>,
    /// Total advance width and line-box height, in pixels.
    pub width: i32,
    pub height: i32,
}

/// Lay out and rasterize `text` at `px` pixels tall.
///
/// Whitespace becomes a fixed gap; codepoints the font has no glyph for are
/// skipped (so unknown input never draws `.notdef` tofu).
pub fn layout(text: &str, px: f32) -> RasterLine {
    let font = font();

    // Line metrics give us the baseline; fall back to rough ratios if absent.
    let (ascent, descent) = match font.horizontal_line_metrics(px) {
        Some(lm) => (lm.ascent, lm.descent),
        None => (px * 0.8, -px * 0.2),
    };
    let height = (ascent - descent).ceil().max(1.0) as i32;
    let baseline = ascent.round() as i32;
    // sitelen words usually sit close together; a < half-em gap reads well.
    let space_adv = px * 0.45;

    let mut glyphs = Vec::new();
    let mut pen = 0.0f32;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pen += space_adv;
            continue;
        }
        if font.lookup_glyph_index(ch) == 0 {
            continue; // no glyph for this codepoint — skip rather than draw tofu
        }
        let (m, bitmap) = font.rasterize(ch, px);
        if m.width > 0 && m.height > 0 {
            let dx = (pen + m.xmin as f32).round() as i32;
            let dy = baseline - (m.ymin + m.height as i32);
            glyphs.push(Glyph { bitmap, w: m.width, h: m.height, dx, dy });
        }
        pen += m.advance_width;
    }

    RasterLine { glyphs, width: pen.ceil().max(0.0) as i32, height }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bundled font must parse and actually rasterize sitelen pona glyphs at
    /// the UCSUR codepoints — otherwise bubbles would render blank at runtime.
    #[test]
    fn rasterizes_sitelen_pona() {
        // 0xF1900 = first UCSUR word glyph ("a"); 0xF196C = "toki".
        let a = char::from_u32(0xF1900).unwrap();
        let toki = char::from_u32(0xF196C).unwrap();
        assert_ne!(font().lookup_glyph_index(a), 0, "font is missing UCSUR glyphs");

        let line = layout(&format!("{toki} {a}"), 24.0);
        assert!(!line.glyphs.is_empty(), "no glyphs rasterized");
        assert!(line.width > 0 && line.height > 0, "empty line box");
        let inked: usize =
            line.glyphs.iter().map(|g| g.bitmap.iter().filter(|&&c| c > 0).count()).sum();
        assert!(inked > 0, "glyph coverage is empty");
    }
}
