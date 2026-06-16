//! bird-panel — a real-time control panel for `desktop-bird`.
//!
//! Connects to a running bird over its Unix control socket and streams
//! [`ControlMsg`]s as the user drags sliders, so the bird's behaviour retunes
//! live. Tuning is session-only: nothing is written to disk; the **Copy values**
//! button exports the current settings as a TOML block to paste into feedback.
//!
//! Run the bird in one terminal (`cargo run --release`) and this in another
//! (`cargo run -p bird-panel --release`).

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

use bird_protocol::{socket_path, ControlMsg, ForceState, Tuning};
use eframe::egui;

/// Built-in style names, in `art::STYLES` order.
const STYLES: [&str; 10] =
    ["sona", "walo", "laso", "jelo", "loje", "suwi", "musi", "lipu", "sitelen", "sewi"];

/// Latin Toki Pona samples (from `demo.py`) for the test-phrase box.
const SAMPLES: [&str; 5] = [
    "toki a",
    "mi moku e kili",
    "sina pona mute",
    "waso lili li tawa sewi",
    "tenpo suno ni li pona",
];

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([360.0, 720.0]).with_title("bird-panel"),
        ..Default::default()
    };
    eframe::run_native("bird-panel", options, Box::new(|_cc| Ok(Box::new(Panel::new()))))
}

struct Panel {
    tuning: Tuning,
    style: String,
    phrase: String,
    say_secs: f32,
    stream: Option<UnixStream>,
    status: String,
}

impl Panel {
    fn new() -> Self {
        let mut p = Panel {
            tuning: Tuning::default(),
            style: STYLES[0].to_string(),
            phrase: SAMPLES[1].to_string(),
            say_secs: 4.0,
            stream: None,
            status: String::new(),
        };
        p.connect();
        p
    }

    /// (Re)connect to the bird's control socket.
    fn connect(&mut self) {
        match UnixStream::connect(socket_path()) {
            Ok(s) => {
                self.stream = Some(s);
                self.status = "connected".to_string();
            }
            Err(e) => {
                self.stream = None;
                self.status = format!("not connected ({e}) — is the bird running?");
            }
        }
    }

    /// Send one control message, dropping the connection on write failure.
    fn send(&mut self, msg: &ControlMsg) {
        let mut line = match serde_json::to_string(msg) {
            Ok(s) => s,
            Err(e) => {
                self.status = format!("encode error: {e}");
                return;
            }
        };
        line.push('\n');
        let result = match self.stream.as_mut() {
            Some(s) => s.write_all(line.as_bytes()).and_then(|_| s.flush()),
            None => {
                self.status = "not connected — click Connect".to_string();
                return;
            }
        };
        if let Err(e) = result {
            self.status = format!("disconnected: {e}");
            self.stream = None;
        } else {
            self.status = "connected".to_string();
        }
    }

    /// Export the current tuning (and style) as a TOML block for feedback.
    fn export_toml(&self) -> String {
        let body = toml::to_string_pretty(&self.tuning).unwrap_or_else(|e| format!("# encode error: {e}"));
        format!("# desktop-bird tuning\nstyle = {:?}\n\n{body}", self.style)
    }
}

impl eframe::App for Panel {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Collected during the frame, then sent once at the end so a drag emits
        // at most one SetTuning per frame.
        let mut tuning_dirty = false;
        // Deferred actions (avoid borrowing self while the closure holds `ui`).
        let mut style_change: Option<String> = None;
        let mut force: Option<ForceState> = None;
        let mut say = false;
        let mut do_connect = false;
        let mut do_copy = false;
        let mut do_reset = false;

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- connection bar ---
            ui.horizontal(|ui| {
                let connected = self.stream.is_some();
                let dot = if connected { "🟢" } else { "🔴" };
                ui.label(format!("{dot} {}", self.status));
            });
            ui.horizontal(|ui| {
                if ui.button(if self.stream.is_some() { "Reconnect" } else { "Connect" }).clicked() {
                    do_connect = true;
                }
                if ui.button("Copy values").clicked() {
                    do_copy = true;
                }
                if ui.button("Reset to defaults").clicked() {
                    do_reset = true;
                }
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                let m = &mut self.tuning.motion;
                ui.collapsing("Movement & timing", |ui| {
                    tuning_dirty |= slider(ui, &mut m.speed, 0.0..=600.0, "speed (px/s)");
                    tuning_dirty |= slider(ui, &mut m.arrive_eps, 0.5..=20.0, "arrive eps (px)");
                    tuning_dirty |= slider(ui, &mut m.follow_eps, 0.0..=300.0, "follow eps (px)");
                    tuning_dirty |= slider(ui, &mut m.flit_delay_min, 0.0..=30.0, "flit delay min (s)");
                    tuning_dirty |= slider(ui, &mut m.flit_delay_max, 0.0..=60.0, "flit delay max (s)");
                    tuning_dirty |= slider(ui, &mut m.flit_radius, 0.0..=400.0, "flit radius (px)");
                    tuning_dirty |= slider(ui, &mut m.follow_flap_linger, 0.0..=2.0, "flap linger (s)");
                    tuning_dirty |= slider64(ui, &mut m.wander_idle_chance, 0.0..=1.0, "wander idle chance");
                    tuning_dirty |= slider(ui, &mut m.wander_idle_min, 0.0..=5.0, "wander idle min (s)");
                    tuning_dirty |= slider(ui, &mut m.wander_idle_max, 0.0..=10.0, "wander idle max (s)");
                    ui.horizontal(|ui| {
                        ui.label("force:");
                        if ui.button("Wander").clicked() {
                            force = Some(ForceState::Wander);
                        }
                        if ui.button("Approach").clicked() {
                            force = Some(ForceState::Approach);
                        }
                        if ui.button("Perch").clicked() {
                            force = Some(ForceState::Perch);
                        }
                        if ui.button("Flit").clicked() {
                            force = Some(ForceState::Flit);
                        }
                    });
                });

                let a = &mut self.tuning.anim;
                ui.collapsing("Animation & appearance", |ui| {
                    egui::ComboBox::from_label("style").selected_text(&self.style).show_ui(ui, |ui| {
                        for name in STYLES {
                            if ui.selectable_value(&mut self.style, name.to_string(), name).clicked() {
                                style_change = Some(self.style.clone());
                            }
                        }
                    });
                    ui.label("frame rate (fps):");
                    tuning_dirty |= slider(ui, &mut a.fps_idle, 0.0..=30.0, "idle");
                    tuning_dirty |= slider(ui, &mut a.fps_fly, 0.0..=30.0, "fly");
                    tuning_dirty |= slider(ui, &mut a.fps_perch, 0.0..=30.0, "perch");
                    tuning_dirty |= slider(ui, &mut a.fps_talk, 0.0..=30.0, "talk");
                    ui.label("pose amplitude (×):");
                    tuning_dirty |= slider(ui, &mut a.wing_amp, 0.0..=2.0, "wing sweep");
                    tuning_dirty |= slider(ui, &mut a.idle_bob, 0.0..=4.0, "idle bob");
                    tuning_dirty |= slider(ui, &mut a.perch_bob, 0.0..=4.0, "perch bob");
                    tuning_dirty |= slider(ui, &mut a.talk_beak, 0.0..=3.0, "talk beak");
                    ui.label("(style/amplitude apply to the procedural bird only)");
                });

                let b = &mut self.tuning.bubble;
                ui.collapsing("Speech bubble", |ui| {
                    tuning_dirty |= slider(ui, &mut b.text_px, 8.0..=64.0, "text size (px)");
                    tuning_dirty |= slideri(ui, &mut b.pad, 0..=24, "padding");
                    tuning_dirty |= slideri(ui, &mut b.tail, 0..=24, "tail");
                    tuning_dirty |= slideri(ui, &mut b.radius, 0..=20, "corner radius");
                    tuning_dirty |= slideru(ui, &mut b.ink, 0..=255, "ink (grey)");
                    ui.separator();
                    ui.label("test phrase (Latin auto-converts; UCSUR sent as-is):");
                    ui.text_edit_singleline(&mut self.phrase);
                    slider(ui, &mut self.say_secs, 0.5..=20.0, "duration (s)");
                    ui.horizontal(|ui| {
                        if ui.button("Say").clicked() {
                            say = true;
                        }
                        for s in SAMPLES {
                            if ui.small_button(s).clicked() {
                                self.phrase = s.to_string();
                                say = true;
                            }
                        }
                    });
                });
            });
        });

        // --- apply deferred actions ---
        if do_connect {
            self.connect();
        }
        if do_reset {
            self.tuning = Tuning::default();
            tuning_dirty = true;
        }
        if do_copy {
            ctx.copy_text(self.export_toml());
            self.status = "copied current values to clipboard".to_string();
        }
        if tuning_dirty {
            let t = self.tuning.clone();
            self.send(&ControlMsg::SetTuning(t));
        }
        if let Some(name) = style_change {
            self.send(&ControlMsg::SetStyle(name));
        }
        if let Some(s) = force {
            self.send(&ControlMsg::Force(s));
        }
        if say {
            let text = to_ucsur(&self.phrase, &mut self.status);
            if !text.trim().is_empty() {
                self.send(&ControlMsg::Bubble { text, secs: self.say_secs });
            }
        }
    }
}

/// f32 slider; returns whether the value changed this frame.
fn slider(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

/// f64 slider (for `wander_idle_chance`).
fn slider64(ui: &mut egui::Ui, v: &mut f64, range: std::ops::RangeInclusive<f64>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

/// i32 slider.
fn slideri(ui: &mut egui::Ui, v: &mut i32, range: std::ops::RangeInclusive<i32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

/// u32 slider.
fn slideru(ui: &mut egui::Ui, v: &mut u32, range: std::ops::RangeInclusive<u32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

/// Convert the test phrase to UCSUR sitelen pona. Latin input (contains ASCII
/// letters) is run through the repo's `sitelen.latin_to_ucsur` via `python3`;
/// already-UCSUR text is sent unchanged. On any failure the raw text is sent and
/// a note is left in `status`.
fn to_ucsur(text: &str, status: &mut String) -> String {
    let text = text.trim();
    if !text.chars().any(|c| c.is_ascii_alphabetic()) {
        return text.to_string(); // already sitelen pona / punctuation
    }
    // Repo root is two levels up from this crate (desktop-bird/panel → repo).
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2);
    let Some(repo) = repo else {
        *status = "couldn't locate repo for transliteration; sent raw".to_string();
        return text.to_string();
    };
    let py = "import sys; sys.path.insert(0, sys.argv[1]); \
              from sitelen import latin_to_ucsur; print(latin_to_ucsur(sys.argv[2]), end='')";
    match std::process::Command::new("python3").arg("-c").arg(py).arg(repo).arg(text).output() {
        Ok(out) if out.status.success() => match String::from_utf8(out.stdout) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                *status = "transliteration gave empty output; sent raw".to_string();
                text.to_string()
            }
        },
        Ok(out) => {
            *status = format!("python3 transliteration failed: {}", String::from_utf8_lossy(&out.stderr).trim());
            text.to_string()
        }
        Err(e) => {
            *status = format!("python3 not available ({e}); sent raw Latin");
            text.to_string()
        }
    }
}
