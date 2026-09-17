"""T800 NPZ diagnostic replay, extending UniLab's motion/replay_npz.py workflow.

Examples (run from the repository root)::

    engineai-replay
    engineai-replay --npz-file old_mujoco.npz --format mujoco
    engineai-replay --check-only --report report.json
    engineai-replay --mode dynamic --headless --steps 100

F8/F9 select joint/body (or use --joint/--body); F10 cycles the diagnostic
layer; F11 switches file/reference (restarts dynamics); F12 toggles plots.
Space pauses; arrows scrub (kinematic); 6/7 go to the previous/next anomaly;
8 restarts. T remains MuJoCo's native transparency toggle, with no replay binding.

Dynamics use nominal XML position PD actuators and reference joint positions,
not a learned policy. Only explicit starts/restarts write simulator state.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf

from unilab.utils.rotation import np_quat_apply_inverse

from engineai_rl_unilab.assets import ASSETS_ROOT, ensure_t800_assets
from engineai_rl_unilab.cli import CONF_ROOT

from .motion_validation import (
    MotionView,
    Tolerances,
    ValidationReport,
    load_views,
    validate_motion,
)


# GLFW key codes, as delivered by mujoco.viewer's key_callback. The callback
# runs AFTER native handling and cannot consume the event. Reserve letters,
# 0..5, and F1..F7 for MuJoCo; in particular K/L are NOT free (skybox/additive).
# Space and arrows are safe here only because this is a passive viewer, whose
# native simulation stepping shortcuts are inactive.
KEY_BINDINGS = {
    32: "pause",  # Space
    262: "next_frame",  # Right
    263: "previous_frame",  # Left
    297: "joint",  # F8
    298: "body",  # F9
    299: "layer",  # F10
    300: "view",  # F11
    301: "plots",  # F12
    ord("6"): "previous_anomaly",
    ord("7"): "next_anomaly",
    ord("8"): "restart",
}
KEY_HELP = (
    "Space pause | F8 joint (--joint) | F9 body (--body)\n"
    "F10 layer | F11 file/reference | F12 plots\n"
    "Kinematic: Left/Right scrub | 6/7 previous/next anomaly\n"
    "8 restart | T transparency (MuJoCo)"
)
LAYERS = ("position", "orientation", "linear", "angular")


def check_viewer_shortcuts():
    """Fail clearly if an installed MuJoCo version reserves one of our keys."""
    native_keys = {
        ord(shortcut[0])
        for _, _, shortcut in (*mujoco.mjVISSTRING, *mujoco.mjRNDSTRING)
        if shortcut
    }
    native_keys.update(range(ord("0"), ord("5") + 1))
    native_keys.update(range(290, 297))  # F1..F7: built-in viewer panels/overlays.
    conflicts = native_keys.intersection(KEY_BINDINGS)
    if conflicts:
        raise RuntimeError(
            f"Replay shortcuts conflict with MuJoCo key codes: {sorted(conflicts)}"
        )


class ReplayState:
    """Viewer-independent stepping, also used for headless regression checks."""

    def __init__(self, model, view: MotionView, mode="kinematic", sim_dt=1 / 150):
        if mode not in ("kinematic", "dynamic"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not np.isfinite(sim_dt) or sim_dt <= 0:
            raise ValueError("sim_dt must be finite and positive")
        self.model, self.view, self.mode = model, view, mode
        self.data = mujoco.MjData(model)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.jnt_bodyid[free[0]] != model.body("LINK_BASE").id:
            raise ValueError("Replay requires exactly one free root LINK_BASE")
        self.root = model.body("LINK_BASE").id
        self.root_qadr = int(model.jnt_qposadr[free[0]])
        self.root_vadr = int(model.jnt_dofadr[free[0]])
        joints = [model.joint(name).id for name in view.joint_names]
        self.qadr, self.vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.actuators = np.array(
            [model.actuator(name).id for name in view.joint_names]
        )
        if mode == "dynamic":
            for joint, actuator in zip(joints, self.actuators):
                if (
                    model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or model.actuator_trnid[actuator, 0] != joint
                    or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or model.actuator_gainprm[actuator, 0] <= 0
                    or not np.isclose(
                        model.actuator_biasprm[actuator, 1],
                        -model.actuator_gainprm[actuator, 0],
                    )
                ):
                    raise ValueError(
                        "Dynamic replay requires named XML position PD actuators"
                    )
        ratio = (1 / view.fps) / sim_dt
        self.substeps = round(ratio)
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError("sim_dt must divide the NPZ frame interval exactly")
        model.opt.timestep = sim_dt
        self.initializations = 0
        self.clipped_targets = 0
        self.frame = 0

    def _controls(self):
        target = self.view.data.joint_pos[self.frame].copy()
        bounds = self.model.actuator_ctrlrange[self.actuators]
        limited = self.model.actuator_ctrllimited[self.actuators].astype(bool)
        clipped = np.clip(target[limited], bounds[limited, 0], bounds[limited, 1])
        self.clipped_targets += int(np.count_nonzero(target[limited] != clipped))
        target[limited] = clipped
        self.data.ctrl[self.actuators] = target

    def set_frame(self, frame: int):
        if not 0 <= frame < len(self.view.data.joint_pos):
            raise ValueError("Frame is outside this clip")
        self.frame = frame
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
        self._controls()
        mujoco.mj_forward(self.model, data)

    def initialize(self, frame=0):
        mujoco.mj_resetData(self.model, self.data)
        self.initializations += 1
        self.set_frame(frame)

    def advance(self) -> bool:
        if self.frame >= len(self.view.data.joint_pos) - 1:
            return False
        if self.mode == "kinematic":
            self.set_frame(self.frame + 1)
        else:
            self._controls()
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
            ):
                raise RuntimeError("Physics produced non-finite state")
            if any(w.number for w in self.data.warning):
                raise RuntimeError("MuJoCo reported a physics warning; replay stopped")
            self.frame += 1
            mujoco.mj_forward(self.model, self.data)
        return True


CYAN = (0.05, 0.85, 1.0, 1.0)
YELLOW = (1.0, 0.8, 0.05, 1.0)
GREEN = (0.2, 1.0, 0.3, 1.0)
RED = (1.0, 0.1, 0.1, 1.0)
GRAY = (0.6, 0.6, 0.6, 0.8)


def _geom(scene, kind, start, end=None, color=CYAN, width=0.008, label=""):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Diagnostic scene capacity exhausted")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.full(3, width),
        np.asarray(start, dtype=float),
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
    )
    if end is not None:
        mujoco.mjv_connector(geom, kind, width, np.asarray(start), np.asarray(end))
    geom.label = label
    scene.ngeom += 1


def draw_diagnostics(
    scene,
    state: ReplayState,
    report: ValidationReport,
    body: int,
    layer="linear",
    vector_scale=0.2,
):
    """Same scale for both arrows; never normalize away magnitude errors."""
    f = state.frame
    view, data = report.view, report.view.data
    for i in range(1, state.model.nbody):
        failed = False
        hotspot = False
        for metric in report.metrics.values():
            if metric.names == view.body_names:
                hotspot |= bool(metric.hotspots[f, i])
                failed |= bool(metric.hotspots[f, i] and metric.failed[i])
        color = RED if failed else YELLOW if hotspot else GREEN
        if not view.pose_comparable[i] or not view.linear_comparable[i]:
            if not hotspot:
                color = GRAY
        _geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            data.body_pos_w[f, i],
            color=color,
            width=0.014 if i == body else 0.008,
            label=view.body_names[i] if i == body else "",
        )
        # Actual simulated body origin (in kinematic mode this is FK).
        _geom(
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            data.body_pos_w[f, i],
            state.data.xpos[i],
            color=GRAY,
            width=2,
        )
    pos = data.body_pos_w[f, body]
    if layer == "position":
        _geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            report.fk.body_pos_w[f, body],
            color=GREEN,
            width=0.02,
            label="FK",
        )
        _geom(
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            pos,
            report.fk.body_pos_w[f, body],
            color=YELLOW,
            width=3,
        )
    elif layer == "orientation":
        for quat, origin, colors in (
            (
                data.body_quat_w[f, body],
                pos,
                [(1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.2, 0.4, 1, 1)],
            ),
            (
                report.fk.body_quat_w[f, body],
                report.fk.body_pos_w[f, body],
                [(1, 0.6, 0.6, 0.65), (0.6, 1, 0.6, 0.65), (0.6, 0.7, 1, 0.65)],
            ),
        ):
            matrix = np.empty(9)
            q = quat.astype(float) / np.linalg.norm(quat)
            mujoco.mju_quat2Mat(matrix, q)
            for axis in range(3):
                _geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    origin,
                    origin + matrix.reshape(3, 3)[:, axis] * 0.15,
                    color=colors[axis],
                    width=0.006,
                )
    else:
        supplied = data.body_lin_vel_w if layer == "linear" else data.body_ang_vel_w
        derived = report.linear_fd if layer == "linear" else report.angular_fd
        for velocity, color in ((supplied[f, body], CYAN), (derived[f, body], YELLOW)):
            if np.isfinite(velocity).all() and np.linalg.norm(velocity) > 1e-10:
                _geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    pos,
                    pos + vector_scale * velocity,
                    color=color,
                )


def _figure(title, series, fps, frame, units):
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title, fig.xlabel = f"{title} [{units}]", "time (s)"
    # The shared color key is in the overlay, leaving room for the curves.
    fig.flg_legend = False
    fig.flg_extend = False
    fig.gridsize[:] = [4, 3]
    fig.xformat, fig.yformat = "%.1f", "%.2g"
    lo, hi = max(0, frame - 100), min(len(series[0][1]), frame + 101)
    # MjvFigure has a fixed point budget. This window is bounded to 201 points.
    values = []
    for i, (name, value, color) in enumerate(series):
        segment = np.asarray(value[lo:hi])
        valid = np.isfinite(segment)
        t = np.arange(lo, hi)[valid] / fps
        y = segment[valid]
        fig.linepnt[i] = len(y)
        fig.linename[i] = f"{name} ({units})"
        fig.linergb[i] = color[:3]
        fig.linedata[i, : 2 * len(y)] = np.column_stack((t, y)).ravel()
        values.extend(y)
    bottom, top = (min(values), max(values)) if values else (0, 1)
    pad = max((top - bottom) * 0.1, 0.01)
    fig.range[:] = [
        [lo / fps, max((hi - 1) / fps, (lo + 1) / fps)],
        [bottom - pad, top + pad],
    ]
    marker = len(series)
    fig.linepnt[marker] = 2
    fig.linename[marker] = "current frame"
    fig.linergb[marker] = (1, 1, 1)
    fig.linedata[marker, :4] = (frame / fps, bottom - pad, frame / fps, top + pad)
    return fig


def diagnostic_figures(report, frame, body, joint, layer, viewport=None):
    data = report.view.data
    if layer in ("linear", "angular"):
        raw = data.body_lin_vel_w if layer == "linear" else data.body_ang_vel_w
        fd = report.linear_fd if layer == "linear" else report.angular_fd
        fk = report.fk.body_lin_vel_w if layer == "linear" else report.fk.body_ang_vel_w
        series = [
            ("stored norm", np.linalg.norm(raw[:, body], axis=-1), CYAN),
            ("FD norm", np.linalg.norm(fd[:, body], axis=-1), YELLOW),
            ("FK norm", np.linalg.norm(fk[:, body], axis=-1), GREEN),
        ]
        units = "m/s" if layer == "linear" else "rad/s"
        comparable = (
            report.view.linear_comparable
            if layer == "linear"
            else report.view.pose_comparable
        )[body]
        suffix = "" if comparable else " [UNVERIFIED]"
        title = f"{report.view.body_names[body].removeprefix('LINK_')} {layer}{suffix}"
    else:
        key = "body_position_fk" if layer == "position" else "body_orientation_fk"
        metric = report.metrics[key]
        series = [
            ("FK error", metric.error[:, body], YELLOW),
            ("limit", np.full(len(data.joint_pos), metric.tolerance), RED),
        ]
        title = report.view.body_names[body].removeprefix("LINK_") + " " + layer
        units = metric.units
    body_fig = _figure(title, series, report.view.fps, frame, units)
    joint_fig = _figure(
        report.view.joint_names[joint] + " velocity",
        [
            ("stored", data.joint_vel[:, joint], CYAN),
            ("FD", report.joint_fd[:, joint], YELLOW),
        ],
        report.view.fps,
        frame,
        "rad/s",
    )
    width = min(460, int(viewport.width * 0.36)) if viewport else 460
    height = min(230, int(viewport.height * 0.24)) if viewport else 230
    left, bottom = (viewport.left, viewport.bottom) if viewport else (0, 0)
    return [
        (mujoco.MjrRect(left, bottom, width, height), body_fig),
        (mujoco.MjrRect(left, bottom + height, width, height), joint_fig),
    ]


def overlay_text(state, report, body, joint, layer, paused):
    frame = state.frame
    lines = [
        f"{report.status} | {report.view.label}",
        f"{state.mode.upper()} | frame {frame}/{len(report.view.data.joint_pos) - 1} "
        f"({frame / report.view.fps:.2f}s) | {'PAUSED' if paused else 'PLAYING'}",
        f"Body: {report.view.body_names[body]} | {layer}",
        "Cyan: stored | Yellow: differentiated | Green: FK",
        "Red: error | Yellow: spike | Gray: unverified",
        "Arrows: shared scale. Plots: vector norms.",
    ]
    if state.mode == "dynamic":
        error = np.linalg.norm(
            state.data.xpos[state.root] - report.view.data.body_pos_w[frame, state.root]
        )
        lines.append(
            f"Nominal XML PD, no policy | root error {error:.3f}m | "
            f"clipped targets {state.clipped_targets}"
        )
    if report.view.notes:
        lines.append("Source non-root pose/COM mapping UNVERIFIED")
    return "\n".join(lines)


def diagnostic_texts(state, report, body, joint, layer, paused, notice=""):
    # MuJoCo limits each overlay column to mjMAXOVERLAY (500) characters.
    labels, values = ["Current-frame errors"], [""]
    for key, metric in report.metrics.items():
        index = joint if metric.names == report.view.joint_names else body
        error = metric.error[state.frame, index]
        labels.append(key)
        values.append(
            f"{error:.4g} {metric.units}"
            if np.isfinite(error)
            else "UNVERIFIED / boundary"
        )
    labels.extend(["Joint", "NPZ position", "NPZ velocity"])
    values.extend(
        [
            report.view.joint_names[joint],
            f"{report.view.data.joint_pos[state.frame, joint]:.4g} rad",
            f"{report.view.data.joint_vel[state.frame, joint]:.4g} rad/s",
        ]
    )
    font = mujoco.mjtFontScale.mjFONTSCALE_100
    return [
        (
            font,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            overlay_text(state, report, body, joint, layer, paused),
            "",
        ),
        (font, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "\n".join(labels), "\n".join(values)),
        (
            font,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            KEY_HELP + (f"\n{notice}" if notice else ""),
            "",
        ),
    ]


def run_viewer(model, reports, args):
    import mujoco.viewer

    check_viewer_shortcuts()
    if not hasattr(mujoco.viewer.Handle, "set_figures"):
        raise RuntimeError(
            "Diagnostic overlays require a MuJoCo viewer with set_figures"
        )
    key = args.view
    report = reports[key]
    state = ReplayState(model, report.view, args.mode, args.sim_dt)
    metric, worst_frame, index = report.worst()
    body = (
        report.view.body_names.index(args.body)
        if args.body
        else (
            index
            if report.metrics[metric].names == report.view.body_names and index
            else 1
        )
    )
    joint = (
        report.view.joint_names.index(args.joint)
        if args.joint
        else (index if report.metrics[metric].names == report.view.joint_names else 0)
    )
    if not args.body and report.metrics[metric].names == report.view.joint_names:
        body = model.jnt_bodyid[model.joint(report.view.joint_names[joint]).id]
    start = (
        args.start_frame
        if args.start_frame is not None
        else (worst_frame if report.failed and args.mode == "kinematic" else 0)
    )
    state.initialize(start)
    paused, plots, layer = not args.play, True, "linear"
    notice = ""
    events = deque()
    print(
        "Opening diagnostic viewer (paused by default). "
        "Failing clips open at their worst frame."
    )
    print(KEY_HELP)
    with mujoco.viewer.launch_passive(
        model, state.data, key_callback=events.append
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
                elif action == "view":
                    key = "reference" if key == "file" else "file"
                    report = reports[key]
                    state.view = report.view
                    state.initialize(state.frame)
                    paused = True
                elif action == "restart":
                    state.initialize(start)
                    paused = True
                elif action == "body":
                    body = body % (model.nbody - 1) + 1
                elif action == "joint":
                    joint = (joint + 1) % len(report.view.joint_names)
                elif action == "plots":
                    plots = not plots
                elif action == "layer":
                    layer = LAYERS[(LAYERS.index(layer) + 1) % len(LAYERS)]
                elif args.mode == "kinematic" and action in (
                    "next_frame",
                    "previous_frame",
                    "next_anomaly",
                    "previous_anomaly",
                ):
                    target = state.frame
                    if action in ("next_frame", "previous_frame"):
                        target += 1 if action == "next_frame" else -1
                    else:
                        frames = report.anomaly_frames
                        if not len(frames):
                            notice = (
                                "No anomaly frames detected in current view; "
                                "playback unchanged."
                            )
                            # No navigation target: do not rewrite state or pause.
                            continue
                        offset = np.searchsorted(frames, state.frame, side="right")
                        if action == "previous_anomaly":
                            offset = np.searchsorted(frames, state.frame) - 1
                        target = frames[offset % len(frames)]
                    state.set_frame(
                        int(np.clip(target, 0, len(report.view.data.joint_pos) - 1))
                    )
                    paused = True
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_diagnostics(
                    viewer.user_scn, state, report, body, layer, args.vector_scale
                )
            viewer.set_texts(
                diagnostic_texts(state, report, body, joint, layer, paused, notice)
            )
            viewer.set_figures(
                diagnostic_figures(
                    report, state.frame, body, joint, layer, viewer.viewport
                )
                if plots
                else []
            )
            viewer.sync()
            if not paused and not state.advance():
                if args.loop:
                    state.initialize(0)
                else:
                    paused = True
            time.sleep(
                max(
                    0.001,
                    (0.05 if paused else 1 / report.view.fps / args.speed)
                    - (time.perf_counter() - t0),
                )
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--npz-file", "--npz_file")
    parser.add_argument("--model-file", "--model_file")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONF_ROOT / "ppo/task/engineai_t800_motion_tracking/mujoco.yaml",
    )
    parser.add_argument("--format", choices=("t800_isaac_v1", "mujoco"))
    parser.add_argument("--view", choices=("file", "reference"), default="reference")
    parser.add_argument("--mode", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--report", type=Path, help="Write both views' diagnostics as JSON"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--sim-dt", type=float)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--vector-scale", type=float, default=0.2)
    parser.add_argument("--body")
    parser.add_argument("--joint")
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
        cfg = OmegaConf.load(args.config)
        base = ASSETS_ROOT.parent
        model_file = (
            Path(args.model_file)
            if args.model_file
            else base
            / (
                cfg.env.commands.motion.get("motion_model_file")
                or cfg.env.scene.model_file
            )
        )
        motion_file = args.npz_file or cfg.env.commands.motion.params.motion_file
        if not isinstance(motion_file, str):
            raise ValueError(
                "Replay inspects one clip at a time; select it with --npz-file"
            )
        path = Path(motion_file) if args.npz_file else base / motion_file
        source_format = (
            args.format or cfg.env.commands.motion.get("motion_adapter") or "mujoco"
        )
        args.sim_dt = args.sim_dt if args.sim_dt is not None else cfg.env.sim_dt
        if args.steps < 0 or any(
            not np.isfinite(v) or v <= 0 for v in (args.speed, args.vector_scale)
        ):
            raise ValueError(
                "steps must be nonnegative; speed and vector-scale must be positive"
            )
        if not args.model_file:
            ensure_t800_assets()
        model, views = load_views(
            path,
            model_file,
            tuple(cfg.env.scene.entities.robot.joint_names),
            source_format,
        )
        if args.body and args.body not in views[args.view].body_names[1:]:
            raise ValueError(f"Unknown body: {args.body}")
        if args.joint and args.joint not in views[args.view].joint_names:
            raise ValueError(f"Unknown joint: {args.joint}")
        if args.start_frame is not None and not 0 <= args.start_frame < len(
            views[args.view].data.joint_pos
        ):
            raise ValueError("start-frame is outside the clip")
        if not np.isclose(views[args.view].fps * cfg.env.ctrl_dt, 1):
            raise ValueError(
                "NPZ fps must match the selected training configuration ctrl_dt"
            )
        tolerances = Tolerances(
            **{name: getattr(args, name + "_tol") for name in vars(Tolerances())}
        )
        reports = {
            key: validate_motion(view, model, tolerances) for key, view in views.items()
        }
        print(f"NPZ: {path}\nModel: {model_file}\nFormat: {source_format}")
        for key, report in reports.items():
            if key == "reference" and views["reference"] is views["file"]:
                continue
            print(report.text())
        print(
            "Schema PASS. Temporal limits use RMS; peak >10x also fails. "
            "Local spikes >3x are highlighted. Root pose/FK is imposed, not independent evidence."
        )
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
            state = ReplayState(model, views[args.view], args.mode, args.sim_dt)
            state.initialize(args.start_frame or 0)
            for _ in range(args.steps):
                if not state.advance():
                    break
            print(
                f"{args.mode}: frame={state.frame}, simulation_time={state.data.time:.6f}, "
                f"initializations={state.initializations}, clipped_targets={state.clipped_targets}"
            )
        else:
            run_viewer(model, reports, args)
        return int(reports[args.view].failed)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Replay error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
