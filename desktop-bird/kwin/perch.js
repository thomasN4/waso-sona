// desktop-bird KWin script (Plasma 6).
//
// KWin exposes no client protocol for window geometry, so this script runs
// *inside* the compositor (QJSEngine sandbox) where it can read the active
// window, and pushes its frameGeometry out to the bird's session-bus service
// via callDBus. The bird (src/kwin.rs) registers org.waso.DesktopBird and
// self-loads this script via org.kde.kwin.Scripting.
//
// frameGeometry is global-logical pixels; for the single-output case that
// equals the bird's output-local buffer coordinates. (Multi-output would need
// the active screen's origin subtracted — out of scope, see RESEARCH.md §5.)

var tracked = null; // the Window we currently follow
var onGeom = null;  // its bound handler, kept so we can disconnect it

function push(w) {
    if (w && !w.deleted && !w.minimized && w.resourceClass !== "desktop-bird") {
        var g = w.frameGeometry;
        callDBus("org.waso.DesktopBird", "/Tracker",
                 "org.waso.DesktopBird.Tracker", "UpdateWindow",
                 Math.round(g.x), Math.round(g.y),
                 Math.round(g.width), Math.round(g.height),
                 w.caption || "", w.resourceClass || "");
    } else {
        callDBus("org.waso.DesktopBird", "/Tracker",
                 "org.waso.DesktopBird.Tracker", "ClearWindow");
    }
}

function untrack() {
    if (tracked && onGeom) {
        try { tracked.frameGeometryChanged.disconnect(onGeom); } catch (e) {}
        try { tracked.interactiveMoveResizeFinished.disconnect(onGeom); } catch (e) {}
        try { tracked.minimizedChanged.disconnect(onGeom); } catch (e) {}
    }
    tracked = null;
    onGeom = null;
}

function track(w) {
    untrack();
    tracked = w;
    if (w) {
        onGeom = function () { push(w); };
        // Re-push on move/resize end and on minimize. frameGeometryChanged may
        // be coalesced during an interactive drag; end-of-drag is fine for a pet.
        try { w.frameGeometryChanged.connect(onGeom); } catch (e) {}
        try { w.interactiveMoveResizeFinished.connect(onGeom); } catch (e) {}
        try { w.minimizedChanged.connect(onGeom); } catch (e) {}
    }
    push(w);
}

// Initial state + follow focus changes (windowActivated passes null when none).
track(workspace.activeWindow);
workspace.windowActivated.connect(track);
