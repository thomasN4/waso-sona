//! Speech-bubble state and the channel used to push text to the bird.
//!
//! Create a linked pair with [`channel`] once at startup:
//! - Keep the [`Sender`] in whatever thread runs the Toki Pona model.
//! - Hand the [`Receiver`] to [`crate::app::AppState`].
//!
//! Send `(text, duration_secs)` on the sender; the bird will display the
//! bubble above its head for the requested duration, then dismiss it.

use std::sync::mpsc::{self, Receiver, Sender};

/// A speech bubble that is currently visible above the bird.
pub struct BubbleState {
    pub text: String,
    pub remaining: f32,
}

impl BubbleState {
    /// Advance the timer by `dt` seconds.  Returns `false` when the bubble
    /// has expired and should be removed.
    pub fn tick(&mut self, dt: f32) -> bool {
        self.remaining -= dt;
        self.remaining > 0.0
    }
}

/// Create a linked `(sender, receiver)` pair for bubble messages.
///
/// The sender is `Clone + Send` and can be handed to the model thread.
/// Each message is `(text, duration_secs)`.
pub fn channel() -> (Sender<(String, f32)>, Receiver<(String, f32)>) {
    mpsc::channel()
}
