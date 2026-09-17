"""Numerical diagnostics consumed by both the NPZ viewer and headless checks.

No data repair happens here. A file view retains the supplied body states; a
reference view uses the exact T800 training loader. Unknown source frame/COM
relationships are excluded explicitly, never silently counted as passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

from unilab.tasks.motion_tracking.common.motion_loader import MotionData, MotionLoader
from unilab.utils.rotation import np_quat_angular_velocity

from .motion_adapter import (
    SOURCE_BODY_NAMES,
    STATE_FIELDS,
    _tracking_fk,
    load_t800_isaac_source,
    read_motion_npz,
)
from .motion_loader import T800MotionLoader


@dataclass
class MotionView:
    label: str
    fps: int
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    data: MotionData
    # Both masks address the target model's body IDs; world is synthetic.
    pose_comparable: np.ndarray
    linear_comparable: np.ndarray
    notes: tuple[str, ...] = ()


def load_views(
    path: str | Path,
    model_file: str | Path,
    joint_names: tuple[str, ...],
    source_format: str,
) -> tuple[mujoco.MjModel, dict[str, MotionView]]:
    """Load one clip. Never concatenate clip boundaries for differentiation."""
    model = mujoco.MjModel.from_xml_path(str(model_file))
    bodies = ("", *(model.body(i).name for i in range(1, model.nbody)))
    comparable = np.ones(model.nbody, dtype=bool)
    comparable[0] = False
    if source_format == "t800_isaac_v1":
        fps, source_joints, arrays = load_t800_isaac_source(path)
        # Same loader as OfficialMotionCommand, including joint mapping and FK.
        loader = T800MotionLoader(
            str(path), model_file=str(model_file), joint_names=joint_names
        )
        reference = MotionView(
            "TRAINING REFERENCE / t800_isaac_v1",
            fps,
            joint_names,
            bodies,
            MotionData(**{f: getattr(loader, f) for f in STATE_FIELDS}),
            comparable.copy(),
            comparable.copy(),
        )
        joint_order = [source_joints.index(name) for name in joint_names]
        body_order = [SOURCE_BODY_NAMES.index(name) for name in bodies[1:]]
        raw = {}
        for key, value in arrays.items():
            if key.startswith("joint_"):
                raw[key] = value[:, joint_order].copy()
            else:
                raw[key] = np.zeros((len(value), model.nbody, value.shape[-1]))
                if key == "body_quat_w":
                    raw[key][:, 0, 0] = 1
                raw[key][:, 1:] = value[:, body_order]
        root_only = np.zeros(model.nbody, dtype=bool)
        root_only[model.body("LINK_BASE").id] = True
        source = MotionView(
            "FILE / Isaac source (partial verification)",
            fps,
            joint_names,
            bodies,
            MotionData(**raw),
            root_only.copy(),
            root_only.copy(),
            (
                "Non-root source pose/FK mapping is UNVERIFIED across models.",
                "Non-root source linear velocity: COM vs origin is UNVERIFIED; "
                "arrows are illustrative, not a validity test.",
                "Source angular differentiation uses world axes (point-independent).",
            ),
        )
        return model, {"file": source, "reference": reference}
    if source_format != "mujoco":
        raise ValueError(f"Unknown source format: {source_format}")
    _, names, body_names, _ = read_motion_npz(path, num_bodies=model.nbody)
    # The legacy training loader indexes columns directly. Do not silently
    # reorder a file for replay and thereby hide a training layout problem.
    if names != joint_names:
        raise ValueError("MuJoCo joint_names must match training joint order exactly")
    if body_names != bodies:
        raise ValueError("MuJoCo body_names must match model body IDs, including world")
    loader = MotionLoader(str(path))
    view = MotionView(
        "FILE / MuJoCo body origins (legacy loader)",
        loader.fps,
        joint_names,
        bodies,
        MotionData(**{f: getattr(loader, f) for f in STATE_FIELDS}),
        comparable.copy(),
        comparable.copy(),
    )
    return model, {"file": view, "reference": view}


@dataclass(frozen=True)
class Tolerances:
    position: float = 0.001  # m, instantaneous FK consistency
    orientation: float = 0.005  # rad
    velocity_fk: float = 0.0001  # m/s or rad/s, instantaneous FK consistency
    joint_velocity: float = 0.05  # rad/s, temporal RMS
    linear_velocity: float = 0.05  # m/s, temporal RMS of vector norm
    angular_velocity: float = 0.15  # rad/s, temporal RMS of vector norm
    joint_limit: float = 0.001  # rad

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("All diagnostic tolerances must be finite and positive")


@dataclass
class Metric:
    name: str
    names: tuple[str, ...]
    units: str
    error: np.ndarray  # (T, entities); NaN means intentionally unverified
    tolerance: float
    temporal: bool = False

    @cached_property
    def verified(self) -> np.ndarray:
        return np.isfinite(self.error).any(axis=0)

    @cached_property
    def rms(self) -> np.ndarray:
        count = np.isfinite(self.error).sum(axis=0)
        return np.sqrt(np.nansum(self.error**2, axis=0) / np.maximum(count, 1))

    @cached_property
    def peak(self) -> np.ndarray:
        return np.max(np.nan_to_num(self.error, nan=0), axis=0)

    @cached_property
    def failed(self) -> np.ndarray:
        # Temporal differences have discretization error. RMS detects sustained
        # disagreement; a 10x peak limit also catches brief severe corruption.
        value = (self.rms > self.tolerance) | (self.peak > 10 * self.tolerance)
        return value if self.temporal else self.peak > self.tolerance

    @cached_property
    def hotspots(self) -> np.ndarray:
        factor = np.where(self.failed, 1, 3)[None, :] if self.temporal else 1
        return self.error > factor * self.tolerance


@dataclass
class ValidationReport:
    view: MotionView
    fk: MotionData
    joint_fd: np.ndarray
    linear_fd: np.ndarray
    angular_fd: np.ndarray
    metrics: dict[str, Metric] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(m.failed.any() for m in self.metrics.values())

    @property
    def complete(self) -> bool:
        return (
            self.view.pose_comparable[1:].all()
            and self.view.linear_comparable[1:].all()
            and len(self.view.data.joint_pos) >= 3
        )

    @property
    def status(self) -> str:
        return "FAIL" if self.failed else "PASS" if self.complete else "PARTIAL"

    @cached_property
    def anomaly_frames(self) -> np.ndarray:
        mask = np.zeros(len(self.view.data.joint_pos), dtype=bool)
        for metric in self.metrics.values():
            mask |= (metric.hotspots & metric.failed[None, :]).any(axis=1)
        return np.flatnonzero(mask)

    def worst(self) -> tuple[str, int, int]:
        candidates = []
        for key, metric in self.metrics.items():
            score = np.nan_to_num(metric.error / metric.tolerance, nan=0)
            if self.failed:
                score[:, ~metric.failed] = 0
            frame, entity = np.unravel_index(np.argmax(score), score.shape)
            candidates.append((float(score[frame, entity]), key, frame, entity))
        _, key, frame, entity = max(candidates)
        return key, int(frame), int(entity)

    def as_dict(self) -> dict:
        metrics = {}
        for key, m in self.metrics.items():
            metrics[key] = {
                "units": m.units,
                "tolerance": m.tolerance,
                "criterion": "RMS or peak > 10x tolerance" if m.temporal else "peak",
                "entities": {
                    name: {
                        "status": "UNVERIFIED"
                        if not m.verified[i]
                        else "FAIL"
                        if m.failed[i]
                        else "PASS",
                        "rms": float(m.rms[i]) if m.verified[i] else None,
                        "peak": float(m.peak[i]) if m.verified[i] else None,
                        "peak_frame": int(np.nanargmax(m.error[:, i]))
                        if m.verified[i]
                        else None,
                    }
                    for i, name in enumerate(m.names)
                    if name
                },
            }
        return {
            "view": self.view.label,
            "status": self.status,
            "complete": bool(self.complete),
            "fps": self.view.fps,
            "frames": len(self.view.data.joint_pos),
            "notes": list(self.view.notes),
            "boundary_policy": "Temporal checks exclude first/last frame; <3 frames unverified",
            "root_fk_policy": "Root pose is imposed; its FK agreement is not independent evidence",
            "anomaly_frames": self.anomaly_frames.tolist(),
            "metrics": metrics,
        }

    def text(self) -> str:
        lines = [f"{self.status}: {self.view.label}"]
        for key, m in self.metrics.items():
            ids = np.flatnonzero(m.failed)
            unverified = sum(not m.verified[i] for i, n in enumerate(m.names) if n)
            lines.append(
                f"  {key}: {len(ids)} failing, {unverified} unverified; "
                f"limit={m.tolerance:g} {m.units}"
            )
            for i in sorted(ids, key=lambda i: -m.rms[i])[:5]:
                frame = int(np.nanargmax(m.error[:, i]))
                lines.append(
                    f"    {m.names[i]} RMS={m.rms[i]:.5g}, peak={m.peak[i]:.5g} "
                    f"{m.units}, frame={frame} ({frame / self.view.fps:.3f}s)"
                )
        lines.extend(f"  NOTE: {note}" for note in self.view.notes)
        lines.append("  Temporal boundaries excluded; frame indices are zero-based.")
        return "\n".join(lines)


def validate_motion(
    view: MotionView, model: mujoco.MjModel, tolerances: Tolerances | None = None
) -> ValidationReport:
    tol = tolerances or Tolerances()
    source = view.data
    n = len(source.joint_pos)
    root = model.body("LINK_BASE").id
    fk_input = {
        f: getattr(source, f)
        if f.startswith("joint_")
        else getattr(source, f)[:, root : root + 1]
        for f in STATE_FIELDS
    }
    fk = _tracking_fk(model, view.joint_names, fk_input)
    joint_fd = np.full(source.joint_vel.shape, np.nan)
    linear_fd = np.full(source.body_lin_vel_w.shape, np.nan)
    angular_fd = np.full(source.body_ang_vel_w.shape, np.nan)
    if n >= 3:
        joint_fd[1:-1] = np.gradient(
            source.joint_pos.astype(float), 1 / view.fps, axis=0
        )[1:-1]
        linear_fd[1:-1] = np.gradient(
            source.body_pos_w.astype(float), 1 / view.fps, axis=0
        )[1:-1]
        for i in range(model.nbody):
            angular_fd[1:-1, i] = np_quat_angular_velocity(
                source.body_quat_w[:, i].astype(float), 1 / view.fps
            )[1:-1]
    report = ValidationReport(view, fk, joint_fd, linear_fd, angular_fd)

    def add(key, units, error, limit, mask=None, temporal=False, joints=False):
        error = np.asarray(error, dtype=float).copy()
        if mask is not None:
            error[:, ~mask] = np.nan
        report.metrics[key] = Metric(
            key,
            view.joint_names if joints else view.body_names,
            units,
            error,
            limit,
            temporal,
        )

    def norm(a):
        return np.linalg.norm(a, axis=-1)

    add(
        "body_position_fk",
        "m",
        norm(source.body_pos_w - fk.body_pos_w),
        tol.position,
        view.pose_comparable,
    )
    q1 = source.body_quat_w.astype(float)
    q2 = fk.body_quat_w.astype(float)
    q1 /= norm(q1)[..., None]
    q2 /= norm(q2)[..., None]
    # q and -q represent the same orientation; chord distance is stable near 0.
    chord = np.minimum(norm(q1 - q2), norm(q1 + q2))
    angle = 4 * np.arcsin(np.clip(chord / 2, 0, 1))
    add("body_orientation_fk", "rad", angle, tol.orientation, view.pose_comparable)
    add(
        "body_linear_fk",
        "m/s",
        norm(source.body_lin_vel_w - fk.body_lin_vel_w),
        tol.velocity_fk,
        view.linear_comparable,
    )
    add(
        "body_angular_fk",
        "rad/s",
        norm(source.body_ang_vel_w - fk.body_ang_vel_w),
        tol.velocity_fk,
        view.pose_comparable,
    )
    add(
        "joint_velocity_fd",
        "rad/s",
        abs(source.joint_vel - joint_fd),
        tol.joint_velocity,
        temporal=True,
        joints=True,
    )
    add(
        "body_linear_fd",
        "m/s",
        norm(source.body_lin_vel_w - linear_fd),
        tol.linear_velocity,
        view.linear_comparable,
        temporal=True,
    )
    all_bodies = np.arange(model.nbody) != 0
    add(
        "body_angular_fd",
        "rad/s",
        norm(source.body_ang_vel_w - angular_fd),
        tol.angular_velocity,
        all_bodies,
        temporal=True,
    )
    ids = [model.joint(name).id for name in view.joint_names]
    limits = model.jnt_range[ids]
    violation = np.maximum(
        np.maximum(limits[:, 0] - source.joint_pos, source.joint_pos - limits[:, 1]), 0
    )
    violation[:, ~model.jnt_limited[ids].astype(bool)] = 0
    add("joint_limits", "rad", violation, tol.joint_limit, joints=True)
    return report
