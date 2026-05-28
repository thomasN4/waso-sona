"""Compare two QLoRA training runs by their TensorBoard scalar logs.

Emits a markdown report to stdout: hyperparameter diff, summary
table (best/final eval_loss, final train_loss/grad_norm, steps
trained, early-stop flag), and the eval/loss curve at matched eval
steps.

Usage::

    python scripts/compare_runs.py                       # defaults
    python scripts/compare_runs.py PREV_DIR NEW_DIR
    python scripts/compare_runs.py PREV_DIR PREV_DIR     # self-test: all Δ = 0
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "training" / "runs"
DEFAULT_PREV = RUNS_DIR / "qlora-20260526T205328Z"

WATCH_TAGS = ("eval/loss", "train/loss", "train/grad_norm", "train/learning_rate")


def _newest_real_run(exclude: Path) -> Path | None:
    candidates = []
    for d in sorted(RUNS_DIR.glob("qlora-*"), reverse=True):
        if d.resolve() == exclude.resolve():
            continue
        cfg = d / "run_config.json"
        if not cfg.exists():
            continue
        if json.loads(cfg.read_text()).get("smoke"):
            continue
        if not (d / "final").exists():
            continue
        candidates.append(d)
    return candidates[0] if candidates else None


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    pattern = str(run_dir / "checkpoints" / "runs" / "*" / "events.out.tfevents.*")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    out: dict[str, list[tuple[int, float]]] = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((s.step, s.value) for s in ea.Scalars(tag))
    for tag, pts in out.items():
        pts.sort(key=lambda p: p[0])
    return out


def _best_min(points: list[tuple[int, float]]) -> tuple[int, float] | None:
    if not points:
        return None
    step, val = min(points, key=lambda p: p[1])
    return step, val


def _fmt_delta(a: float | None, b: float | None, fmt: str = ".4f") -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:{fmt}}"


def _config_diff(prev_cfg: dict, new_cfg: dict) -> list[tuple[str, str, str]]:
    keys = sorted(set(prev_cfg) | set(new_cfg))
    diffs = []
    for k in keys:
        pv, nv = prev_cfg.get(k, "<missing>"), new_cfg.get(k, "<missing>")
        if pv != nv:
            diffs.append((k, str(pv), str(nv)))
    return diffs


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    aligned_rows = [headers, ["---"] * len(headers)] + rows
    return "\n".join("| " + " | ".join(r) + " |" for r in aligned_rows)


def compare(prev_dir: Path, new_dir: Path) -> str:
    prev_cfg = json.loads((prev_dir / "run_config.json").read_text())
    new_cfg = json.loads((new_dir / "run_config.json").read_text())

    prev_sc = _load_scalars(prev_dir)
    new_sc = _load_scalars(new_dir)

    lines = []
    lines.append(f"# Comparison: `{prev_dir.name}` → `{new_dir.name}`\n")

    # Config diff -----------------------------------------------------------
    diffs = _config_diff(prev_cfg, new_cfg)
    lines.append("## Hyperparameter diff\n")
    if not diffs:
        lines.append("_No config differences — only dataset content changed._\n")
    else:
        lines.append(_markdown_table(
            ["knob", "prev", "new"],
            [[k, pv, nv] for k, pv, nv in diffs],
        ) + "\n")

    # Summary ---------------------------------------------------------------
    prev_eval = prev_sc.get("eval/loss", [])
    new_eval = new_sc.get("eval/loss", [])
    prev_tloss = prev_sc.get("train/loss", [])
    new_tloss = new_sc.get("train/loss", [])
    prev_gn = prev_sc.get("train/grad_norm", [])
    new_gn = new_sc.get("train/grad_norm", [])

    def _last(pts):
        return pts[-1] if pts else (None, None)

    def _steps_reached(pts):
        return pts[-1][0] if pts else 0

    prev_best = _best_min(prev_eval)
    new_best = _best_min(new_eval)
    prev_last_eval = _last(prev_eval)
    new_last_eval = _last(new_eval)
    prev_last_tl = _last(prev_tloss)
    new_last_tl = _last(new_tloss)
    prev_last_gn = _last(prev_gn)
    new_last_gn = _last(new_gn)
    prev_steps = _steps_reached(prev_tloss)
    new_steps = _steps_reached(new_tloss)
    prev_early = prev_cfg.get("max_steps", 0) > prev_steps
    new_early = new_cfg.get("max_steps", 0) > new_steps

    lines.append("## Summary\n")
    rows = [
        [
            "best eval/loss",
            f"{prev_best[1]:.4f} @ step {prev_best[0]}" if prev_best else "—",
            f"{new_best[1]:.4f} @ step {new_best[0]}" if new_best else "—",
            _fmt_delta(
                prev_best[1] if prev_best else None,
                new_best[1] if new_best else None,
            ),
        ],
        [
            "final eval/loss",
            f"{prev_last_eval[1]:.4f}" if prev_last_eval[1] is not None else "—",
            f"{new_last_eval[1]:.4f}" if new_last_eval[1] is not None else "—",
            _fmt_delta(prev_last_eval[1], new_last_eval[1]),
        ],
        [
            "final train/loss",
            f"{prev_last_tl[1]:.4f}" if prev_last_tl[1] is not None else "—",
            f"{new_last_tl[1]:.4f}" if new_last_tl[1] is not None else "—",
            _fmt_delta(prev_last_tl[1], new_last_tl[1]),
        ],
        [
            "final train/grad_norm",
            f"{prev_last_gn[1]:.4f}" if prev_last_gn[1] is not None else "—",
            f"{new_last_gn[1]:.4f}" if new_last_gn[1] is not None else "—",
            _fmt_delta(prev_last_gn[1], new_last_gn[1]),
        ],
        [
            "steps trained",
            f"{prev_steps} / {prev_cfg.get('max_steps')}",
            f"{new_steps} / {new_cfg.get('max_steps')}",
            "—",
        ],
        [
            "early-stopped",
            "yes" if prev_early else "no",
            "yes" if new_early else "no",
            "—",
        ],
    ]
    lines.append(_markdown_table(["metric", "prev", "new", "Δ"], rows) + "\n")

    # eval/loss curve at matched steps -------------------------------------
    lines.append("## eval/loss curve\n")
    if not prev_eval and not new_eval:
        lines.append("_no eval/loss scalars found in either run_\n")
    else:
        steps = sorted(set([s for s, _ in prev_eval] + [s for s, _ in new_eval]))
        prev_map = dict(prev_eval)
        new_map = dict(new_eval)
        curve_rows = []
        for s in steps:
            pv = prev_map.get(s)
            nv = new_map.get(s)
            curve_rows.append([
                str(s),
                f"{pv:.4f}" if pv is not None else "—",
                f"{nv:.4f}" if nv is not None else "—",
                _fmt_delta(pv, nv),
            ])
        lines.append(
            _markdown_table(["step", "prev", "new", "Δ"], curve_rows) + "\n"
        )

    # Missing tags ----------------------------------------------------------
    missing_prev = [t for t in WATCH_TAGS if t not in prev_sc]
    missing_new = [t for t in WATCH_TAGS if t not in new_sc]
    if missing_prev or missing_new:
        lines.append("## Missing tags\n")
        if missing_prev:
            lines.append(f"- prev: {', '.join(missing_prev)}")
        if missing_new:
            lines.append(f"- new: {', '.join(missing_new)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "prev", nargs="?", default=str(DEFAULT_PREV),
        help=f"prior run dir (default: {DEFAULT_PREV.name})",
    )
    ap.add_argument(
        "new", nargs="?", default=None,
        help="new run dir (default: newest non-smoke run with final/)",
    )
    args = ap.parse_args(argv)

    prev = Path(args.prev)
    if not prev.is_absolute():
        prev = (REPO_ROOT / prev).resolve()
    if not (prev / "run_config.json").exists():
        print(f"prev run not found: {prev}", file=sys.stderr)
        return 1

    if args.new is None:
        new = _newest_real_run(prev)
        if new is None:
            print(
                "no non-smoke run with final/ found other than prev; "
                "pass NEW_DIR explicitly",
                file=sys.stderr,
            )
            return 1
    else:
        new = Path(args.new)
        if not new.is_absolute():
            new = (REPO_ROOT / new).resolve()
    if not (new / "run_config.json").exists():
        print(f"new run not found: {new}", file=sys.stderr)
        return 1

    print(compare(prev, new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
