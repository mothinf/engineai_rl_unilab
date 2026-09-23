"""T800 NPZ replay: pose-driven robot plus one cyan body-velocity skeleton.

Examples (from the repository root)::

    engineai-replay --format mujoco --npz-file motion.npz --play --loop
    engineai-replay --ghost-window 0.2
    engineai-replay --check-only --report report.json

F8 toggles the body-velocity skeleton. F12 selects a body
(or use --body). Space pauses, Left/Right scrub, 6/7 navigate anomalies,
8 restarts. C/F/T retain MuJoCo's contact-point/contact-force/transparency toggles.

Ghosts are kinematic reconstructions, NOT physical simulations. Recorded
states are never repaired. Unknown source frame/COM mappings remain UNVERIFIED.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf

from unilab.utils.rotation import np_quat_apply_inverse

from engineai_rl_unilab.assets import ASSETS_ROOT, ensure_t800_assets
from engineai_rl_unilab.cli import CONF_ROOT

from .motion_ghosts import GHOST_NAME, GhostOptions, build_ghost
from .motion_validation import Tolerances, load_views, validate_motion


# Native handling runs BEFORE Python's callback; never reuse native shortcuts.
# Space/arrows are safe only in this passive (non-simulating) viewer.
KEY_BINDINGS = {
    32: "pause",
    262: "next_frame",
    263: "previous_frame",
    297: "ghost",
    301: "body",
    ord("6"): "previous_anomaly",
    ord("7"): "next_anomaly",
    ord("8"): "restart",
}
KEY_HELP = (
    "Space pause | Arrows scrub | 6/7 anomalies\n"
    "F8 cyan skeleton on/off\n"
    "F12 body (--body)\n"
    "8 restart | C contacts / F forces / T transparency (native)"
)
GHOST_COLOR = (0.0, 1.0, 1.0, 1.0)
GHOST_RADIUS = 0.007  # 7 mm radius; solid, thicker than the former dashed skeleton.
GHOST_NODE_RADIUS = 0.014  # Small spheres distinguish nodes from the thick bones.


def check_viewer_shortcuts():
    native = {
        ord(shortcut[0])
        for _, _, shortcut in (*mujoco.mjVISSTRING, *mujoco.mjRNDSTRING)
        if shortcut
    }
    native.update(range(ord("0"), ord("5") + 1))
    native.update(range(290, 297))
    conflicts = native.intersection(KEY_BINDINGS)
    if conflicts:
        raise RuntimeError(
            f"Replay shortcuts conflict with MuJoCo key codes: {sorted(conflicts)}"
        )


def enqueue_replay_key(events, key):
    """Native keys are already handled by MuJoCo: never toggle them again."""
    if key in KEY_BINDINGS:
        events.append(key)


def contact_status(state, option):
    """Explain native visibility without changing flags or inventing contacts."""
    points = bool(option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
    forces = bool(option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
    lines = [
        f"C contacts: {'ON' if points else 'off'} | F forces: {'ON' if forces else 'off'} | ncon={state.data.ncon}"
    ]
    if (points or forces) and not state.data.ncon:
        lines.append("No contacts at this frame; no contact graphics.")
    elif forces:
        lines.append("Forces: MuJoCo pose solve, NOT recorded NPZ forces.")
    return "\n".join(lines)


def warn_motion_report(path, report):
    """Emit once per file, before opening a window; retain machine exit codes."""

    def warn(message):
        print(f"[WARNING] {message}", file=sys.stderr, flush=True)

    if report.failed or report.warnings:
        frame, body = report.worst()
        if report.failed:
            warn(f"NPZ model contract check FAILED: {path}")
        if report.warnings:
            warn(f"NPZ temporal consistency diagnostic thresholds exceeded: {path}")
            warn(
                "Sampling/integration diagnostics are warnings, not proof of invalid data."
            )
        warn(
            f"{len(report.anomaly_frames)} anomalous frames; worst frame={frame} "
            f"({frame / report.view.fps:.3f}s), body={report.view.body_names[body]}"
        )
        failed_metrics = [
            key
            for key, metric in report.base.metrics.items()
            if not metric.temporal and metric.failed.any()
        ]
        if failed_metrics:
            warn("Failed numerical checks: " + ", ".join(failed_metrics))
        temporal_metrics = [
            key
            for key, metric in report.base.metrics.items()
            if metric.temporal and metric.failed.any()
        ]
        if temporal_metrics:
            warn("Temporal diagnostic warnings: " + ", ".join(temporal_metrics))
        name, ghost = GHOST_NAME, report.ghost
        for label, error, limit, scale, unit in (
            ("position", ghost.position_error, ghost.position_tol, 100, "cm"),
            ("angle", ghost.angle_error, ghost.angle_tol, 180 / np.pi, "deg"),
        ):
            if not np.any(error > limit):
                continue
            f, b = np.unravel_index(np.nanargmax(error), error.shape)
            warn(
                f"{name} {label}: body={report.view.body_names[b]}, frame={f}, "
                f"peak={error[f, b] * scale:.3f}{unit}, limit={limit * scale:g}{unit}"
            )
        warn("Replay remains available for diagnosis; use 6/7 to inspect anomalies.")
    if not report.complete:
        warn(
            f"NPZ verification INCOMPLETE: {path}; unknown mapping/fields or "
            "insufficient history. Unverified data is not a PASS."
        )


class ReplayState:
    """Random-access pose replay. No mj_step, actuator control or integration."""

    def __init__(self, model, view):
        self.model, self.view = model, view
        self.data = mujoco.MjData(model)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.root = model.body("LINK_BASE").id
        if len(free) != 1 or model.jnt_bodyid[free[0]] != self.root:
            raise ValueError("Replay requires exactly one free root LINK_BASE")
        self.root_qadr = int(model.jnt_qposadr[free[0]])
        self.root_vadr = int(model.jnt_dofadr[free[0]])
        joints = [model.joint(name).id for name in view.joint_names]
        self.qadr, self.vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.frame = 0

    def set_frame(self, frame):
        if not 0 <= frame < len(self.view.data.joint_pos):
            raise ValueError("Frame is outside this clip")
        self.frame = int(frame)
        motion, data = self.view.data, self.data
        q, v = self.root_qadr, self.root_vadr
        quat = motion.body_quat_w[frame, self.root].astype(float).copy()
        quat /= np.linalg.norm(quat)
        data.qpos[q : q + 3] = motion.body_pos_w[frame, self.root]
        data.qpos[q + 3 : q + 7] = quat
        data.qpos[self.qadr] = motion.joint_pos[frame]
        data.qvel[v : v + 3] = motion.body_lin_vel_w[frame, self.root]
        data.qvel[v + 3 : v + 6] = np_quat_apply_inverse(
            quat[None], motion.body_ang_vel_w[frame, self.root][None]
        )[0]
        data.qvel[self.vadr] = motion.joint_vel[frame]
        data.time = frame / self.view.fps
        mujoco.mj_forward(self.model, data)

    def advance(self):
        if self.frame >= len(self.view.data.joint_pos) - 1:
            return False
        self.set_frame(self.frame + 1)
        return True


def draw_ghost(scene, state, report, enabled):
    """Append cyan bones and spherical body-origin nodes; no axes or error lines."""
    if not enabled:
        return
    ghost, f = report.ghost, state.frame
    for body in range(1, state.model.nbody):
        if not ghost.drawable[f, body]:
            continue
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Ghost scene capacity exhausted")
        node = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            node,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, GHOST_NODE_RADIUS),
            ghost.position[f, body],
            np.eye(3).ravel(),
            np.asarray(GHOST_COLOR, dtype=np.float32),
        )
        node.label = ""
        scene.ngeom += 1
        parent = state.model.body_parentid[body]
        if not parent or not (ghost.drawable[f, body] and ghost.drawable[f, parent]):
            continue
        start, end = ghost.position[f, parent], ghost.position[f, body]
        if np.linalg.norm(end - start) < 1e-8:
            continue
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Ghost scene capacity exhausted")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.full(3, GHOST_RADIUS),
            start,
            np.eye(3).ravel(),
            np.asarray(GHOST_COLOR, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            GHOST_RADIUS,
            start,
            end,
        )
        geom.label = ""
        scene.ngeom += 1


def ghost_texts(state, report, enabled, body, paused, notice="", native_status=""):
    f = state.frame
    n = len(report.view.data.joint_pos)
    lines = [
        f"{report.status} | {report.view.label.split(' (')[0]}",
        f"frame {f}/{n - 1} ({f / report.view.fps:.2f}s) | {'PAUSED' if paused else 'PLAYING'}",
        f"Window {report.window_frames / report.view.fps:.3f}s ({report.window_frames} frames) | scale 1:1",
        f"Velocity limits {report.options.position_tol * 100:g}cm / {report.options.angle_tol_deg:g}deg",
        "Cyan skeleton: integrated body velocity",
        "Rotation/joint errors: see numeric checks",
        "Pose-only replay; no physics stepping",
    ]
    if report.view.notes:
        lines.append("Source frame/COM mapping UNVERIFIED")
    if report.unsupported_fields:
        lines.append(
            f"Unsupported NPZ fields: {len(report.unsupported_fields)} (see report)"
        )
    if native_status:
        lines.extend(native_status.splitlines())
    labels = [report.view.body_names[body]]
    values = ["position / angle vs main FK"]
    ghost = report.ghost
    labels.append("body-velocity" + (" [on]" if enabled else " [off]"))
    if f < report.window_frames:
        value = f"WARM-UP: need frame {report.window_frames}"
    elif not ghost.drawable[f, body]:
        value = "UNVERIFIED frame/COM mapping"
    else:
        value = f"{ghost.position_error[f, body] * 100:.2f}cm / {np.rad2deg(ghost.angle_error[f, body]):.2f}deg"
    values.append(value)
    if not report.complete:
        labels.append("Coverage")
        values.append("PARTIAL: see terminal/JSON")
    font = mujoco.mjtFontScale.mjFONTSCALE_100
    return [
        (font, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(lines), ""),
        (font, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "\n".join(labels), "\n".join(values)),
        (
            font,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            KEY_HELP + (f"\n{notice}" if notice else ""),
            "",
        ),
    ]


def ghost_peak(report, frame):
    """Maximum comparable non-world body position residual in metres."""
    valid = report.ghost.drawable[frame].copy()
    valid[0] = False
    ids = np.flatnonzero(valid)
    if frame < report.window_frames or not len(ids):
        return -1.0, None
    body = int(ids[np.argmax(report.ghost.position_error[frame, ids])])
    return float(report.ghost.position_error[frame, body]), body


def ghost_error_log(report, frame):
    """One position metric only; this is not an overall NPZ pass verdict."""
    error, body = ghost_peak(report, frame)
    prefix = "[NPZ drift]"
    if frame < report.window_frames:
        return f"{prefix}[等待] frame={frame} 历史不足，暂不计算位置偏差"
    if body is None:
        return f"{prefix}[UNVERIFIED] frame={frame} 无可比较部件"
    tol = report.options.position_tol
    complete = np.all(report.ghost.drawable[frame, 1:])
    status = "超限" if error > tol else "位置偏差正常" if complete else "UNVERIFIED"
    coverage = "" if complete else " | 仅部分部件可比较，不能判定整体正常"
    return (
        f"{prefix}[{status}] 最大位置偏差={error * 1000:.1f} mm "
        f"| 阈值={tol * 1000:.1f} mm | 部件={report.view.body_names[body]} "
        f"| frame={frame} ({frame / report.view.fps:.2f}s) "
        f"| 积分窗口={report.window_frames / report.view.fps:.3f}s{coverage}"
    )


class GhostErrorLogger:
    """At most one line/second; retain the peak of all newly displayed frames."""

    def __init__(self, report):
        self.report = report
        self.last_frame = None
        self.last_time = None
        self.pending = None

    def update(self, frame, now):
        if frame != self.last_frame:
            candidate = (ghost_peak(self.report, frame)[0], frame)
            if self.pending is None or candidate[0] >= self.pending[0]:
                self.pending = candidate
            self.last_frame = frame
        if self.pending is not None and (
            self.last_time is None or now - self.last_time >= 1.0
        ):
            message = ghost_error_log(self.report, self.pending[1])
            self.pending = None
            self.last_time = now
            return message
        return None


def run_viewer(model, reports, args):
    import mujoco.viewer

    check_viewer_shortcuts()
    key = args.view
    report = reports[key]
    state = ReplayState(model, report.view)
    worst_frame, worst_body = report.worst()
    body = report.view.body_names.index(args.body) if args.body else worst_body
    start = (
        args.start_frame
        if args.start_frame is not None
        else worst_frame
        if report.failed or report.warnings
        else 0
    )
    state.set_frame(start)
    paused, notice = not args.play, ""
    enabled = not args.no_ghost
    error_logger = None if args.no_error_log else GhostErrorLogger(report)
    events = deque()
    print(
        "Opening ghost viewer (paused by default); clips with errors/warnings start at their worst frame."
    )
    print(KEY_HELP)
    if error_logger is not None:
        print(
            "[NPZ drift] 每秒最多一条：本次输出间隔内经过帧的最大位置偏差；不是整份 NPZ 的合格结论。"
        )
    with mujoco.viewer.launch_passive(
        model, state.data, key_callback=lambda key: enqueue_replay_key(events, key)
    ) as viewer:
        viewer.cam.lookat[:] = state.data.xpos[state.root]
        viewer.cam.distance = 3.5
        viewer.cam.azimuth, viewer.cam.elevation = 135, -15
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        while viewer.is_running():
            t0 = time.perf_counter()
            while events:
                action = KEY_BINDINGS.get(events.popleft())
                if action is not None:
                    notice = ""
                if action == "pause":
                    paused = not paused
                elif action == "ghost":
                    enabled = not enabled
                elif action == "body":
                    body = body % (model.nbody - 1) + 1
                elif action == "restart":
                    state.set_frame(start)
                    paused = True
                elif action in (
                    "next_frame",
                    "previous_frame",
                    "next_anomaly",
                    "previous_anomaly",
                ):
                    if action in ("next_frame", "previous_frame"):
                        target = state.frame + (1 if action == "next_frame" else -1)
                    else:
                        frames = report.anomaly_frames
                        if not len(frames):
                            notice = "No anomaly frames detected in current view; playback unchanged."
                            continue
                        offset = np.searchsorted(frames, state.frame, side="right")
                        if action == "previous_anomaly":
                            offset = np.searchsorted(frames, state.frame) - 1
                        target = frames[offset % len(frames)]
                    state.set_frame(
                        int(np.clip(target, 0, len(report.view.data.joint_pos) - 1))
                    )
                    paused = True
            if error_logger is not None:
                error_message = error_logger.update(state.frame, time.monotonic())
                if error_message is not None:
                    print(error_message, flush=True)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_ghost(viewer.user_scn, state, report, enabled)
                native_status = contact_status(state, viewer.opt)
            viewer.set_texts(
                ghost_texts(state, report, enabled, body, paused, notice, native_status)
            )
            viewer.sync()
            if not paused and not state.advance():
                if args.loop:
                    state.set_frame(0)
                else:
                    paused = True
            time.sleep(
                max(
                    0.001,
                    (0.05 if paused else 1 / report.view.fps / args.speed)
                    - (time.perf_counter() - t0),
                )
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--npz-file", "--npz_file")
    parser.add_argument("--model-file", "--model_file")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONF_ROOT / "ppo/task/engineai_t800_motion_tracking/mujoco.yaml",
    )
    parser.add_argument(
        "--format", choices=("t800_isaac_v1", "mujoco"), default="mujoco"
    )
    parser.add_argument(
        "--view",
        choices=("file",),
        default="file",
        help="Only recorded file data is inspected; convert Isaac data offline first for training",
    )
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="Hide the cyan skeleton; checks still run",
    )
    parser.add_argument(
        "--no-error-log",
        action="store_true",
        help="Disable periodic position-error logs; validation and warnings remain enabled",
    )
    parser.add_argument(
        "--ghost-window",
        type=float,
        default=0.2,
        help="Full history window in seconds (rounded to NPZ frames)",
    )
    parser.add_argument(
        "--ghost-position-tol",
        type=float,
        default=0.03,
        help="Velocity ghost position threshold, metres",
    )
    parser.add_argument(
        "--ghost-angle-tol-deg",
        type=float,
        default=5,
        help="Velocity ghost angle threshold, degrees",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check without replay: exit 1 for model contract failures, 2 for load errors; temporal warnings exit 0",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Exercise pose replay without opening a window (no physics)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write numerical and ghost diagnostics for the selected file",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--body")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--play", action="store_true", help="Start playing instead of paused"
    )
    for name, value in vars(Tolerances()).items():
        parser.add_argument(
            "--" + name.replace("_", "-") + "-tol", type=float, default=value
        )
    args = parser.parse_args(argv)
    try:
        options = GhostOptions(
            args.ghost_window, args.ghost_position_tol, args.ghost_angle_tol_deg
        )
        if args.steps < 0 or not np.isfinite(args.speed) or args.speed <= 0:
            raise ValueError(
                "steps must be nonnegative; speed must be finite and positive"
            )
        cfg = OmegaConf.load(args.config)
        base = ASSETS_ROOT.parent
        model_file = (
            Path(args.model_file)
            if args.model_file
            else base / cfg.env.scene.model_file
        )
        motion_file = args.npz_file or cfg.env.commands.motion.params.motion_file
        if not isinstance(motion_file, str):
            raise ValueError(
                "Replay inspects one clip at a time; select it with --npz-file"
            )
        path = Path(motion_file) if args.npz_file else base / motion_file
        source_format = args.format
        if not args.model_file:
            ensure_t800_assets()
        model, views = load_views(
            path,
            model_file,
            tuple(cfg.env.scene.entities.robot.joint_names),
            source_format,
        )
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
        if args.body and args.body not in views[args.view].body_names[1:]:
            raise ValueError(f"Unknown body: {args.body}")
        if args.start_frame is not None and not 0 <= args.start_frame < len(
            views[args.view].data.joint_pos
        ):
            raise ValueError("start-frame is outside the clip")
        tolerances = Tolerances(
            **{name: getattr(args, name + "_tol") for name in vars(Tolerances())}
        )
        reports, cache = {}, {}
        for key, view in views.items():
            if id(view) not in cache:
                cache[id(view)] = build_ghost(
                    validate_motion(view, model, tolerances), model, options, fields
                )
            reports[key] = cache[id(view)]
        print(
            f"NPZ: {path}\nModel: {model_file}\nFormat: {source_format}\nSelected view: {args.view}"
        )
        for key, report in reports.items():
            print(report.base.text())
            print(
                f"Ghost result {report.status}; coverage {'complete' if report.complete else 'PARTIAL'}; window {report.window_frames / report.view.fps:g}s"
            )
            name, ghost = GHOST_NAME, report.ghost
            valid = np.isfinite(ghost.position_error)
            if valid.any():
                print(
                    f"  {name}: peak {np.nanmax(ghost.position_error) * 100:.3f}cm / {np.rad2deg(np.nanmax(ghost.angle_error)):.3f}deg"
                )
            else:
                print(
                    f"  {name}: UNVERIFIED (mapping, shared root, or insufficient history)"
                )
            if report.unsupported_fields:
                print(
                    f"  Unsupported NPZ fields: {', '.join(report.unsupported_fields)}"
                )
        print(
            "The cyan skeleton needs a full window; rotation and joint errors may only appear in numeric checks.\nConverted-file checks do NOT certify original source fields. Overlap does NOT prove physical feasibility."
        )
        warn_motion_report(path, reports[args.view])
        if args.report:
            args.report.write_text(
                json.dumps(
                    {k: r.as_dict() for k, r in reports.items()},
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
        if args.check_only:
            return int(reports[args.view].failed)
        if args.headless:
            state = ReplayState(model, views[args.view])
            state.set_frame(args.start_frame or 0)
            for _ in range(args.steps):
                if not state.advance():
                    if args.loop:
                        state.set_frame(0)
                    else:
                        break
            print(
                f"Pose-only replay: frame={state.frame}, time={state.data.time:.6f}s (no physics)"
            )
        else:
            run_viewer(model, reports, args)
        return int(reports[args.view].failed)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Replay error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
