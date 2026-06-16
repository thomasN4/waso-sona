//! Live control socket: receives [`ControlMsg`]s from the bird-panel.
//!
//! A background thread binds a Unix socket and reads newline-delimited JSON
//! `ControlMsg`s, forwarding each over an mpsc channel that the main Wayland
//! loop drains every frame in `AppState::tick`. Same shape as the stdin bubble
//! reader (`main.rs`) and the KWin D-Bus bridge (`kwin.rs`): a thread feeds a
//! channel; the render loop never blocks. One-directional (panel → bird).

use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::sync::mpsc::{self, Receiver, Sender};

use bird_protocol::{socket_path, ControlMsg};

/// Start listening for control messages. Returns a receiver the render loop
/// drains each frame; on bind failure the bird simply runs untunable (the
/// receiver stays empty) rather than refusing to start.
pub fn start() -> Receiver<ControlMsg> {
    let (tx, rx) = mpsc::channel();
    let path = socket_path();
    // A stale socket from a previous run would make bind fail with EADDRINUSE.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(err) => {
            eprintln!("desktop-bird: control socket bind failed ({err}); live tuning disabled");
            return rx;
        }
    };
    eprintln!("desktop-bird: control socket listening at {}", path.display());
    std::thread::spawn(move || serve(listener, tx));
    rx
}

/// Accept clients one at a time; for each, parse newline-delimited JSON
/// `ControlMsg`s and forward them. Loops back to `accept` when a client
/// disconnects, so the panel can be closed and reopened freely.
fn serve(listener: UnixListener, tx: Sender<ControlMsg>) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let reader = BufReader::new(stream);
        for line in reader.lines() {
            let Ok(line) = line else { break }; // client closed / read error
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<ControlMsg>(line) {
                Ok(msg) => {
                    if tx.send(msg).is_err() {
                        return; // render loop gone; stop serving
                    }
                }
                Err(err) => eprintln!("desktop-bird: ignoring bad control message: {err}"),
            }
        }
    }
}
