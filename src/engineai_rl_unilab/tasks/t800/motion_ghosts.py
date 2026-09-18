"""Independent, time-aligned motion reconstructions; no physics or data repair."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .motion_npz import STATE_FIELDS, tracking_fk
from .motion_validation import ValidationReport


GHOST_NAMES = ("pose", "body-velocity", "joint-velocity")
KNOWN_FIELDS = frozenset((*STATE_FIELDS, "fps", "joint_names", "body_names"))


def quat_mul(a, b):
    """Hamilton product of broadcastable wxyz arrays."""
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate(
        (
            aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
            aw * bv + bw * av + np.cross(av, bv),
        ),
        axis=-1,
    )


def orientation_error(a, b):
    """Geodesic angle, stable near zero and invariant to quaternion sign."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return 4 * np.arcsin(np.clip(chord / 2, 0, 1))


def integrate_linear(position, velocity, steps, dt):
    """Every endpoint uses its own recorded anchor and a full trapezoid window."""
    out = np.full(position.shape, np.nan, dtype=float)
    n = len(position)
    if steps >= n:
        return out
    # Sum each window independently: neither playback order nor an earlier
    # window's error influences the next reconstruction.
    out[steps:] = position[:-steps]
    for offset in range(steps):
        out[steps:] += (
            0.5
            * dt
            * (
                velocity[offset : n - steps + offset]
                + velocity[offset + 1 : n - steps + offset + 1]
            )
        )
    return out


def integrate_world_orientation(quaternion, omega, steps, dt):
    out = np.full(quaternion.shape, np.nan, dtype=float)
    n = len(quaternion)
    if steps >= n:
        return out
    out[steps:] = quaternion[:-steps]
    for offset in range(steps):
        rotation = (
            0.5
            * dt
            * (
                omega[offset : n - steps + offset]
                + omega[offset + 1 : n - steps + offset + 1]
            )
        )
        theta = np.linalg.norm(rotation, axis=-1, keepdims=True)
        delta = np.concatenate(
            (np.cos(theta / 2), 0.5 * np.sinc(theta / (2 * np.pi)) * rotation), axis=-1
        )
        # World-frame angular velocity acts on the LEFT, not in local axes.
        out[steps:] = quat_mul(delta, out[steps:])
        out[steps:] /= np.linalg.norm(out[steps:], axis=-1, keepdims=True)
    return out


@dataclass(frozen=True)
class GhostOptions:
    window: float = 0.2
    position_tol: float = 0.03
    angle_tol_deg: float = 5.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Ghost window and tolerances must be finite and positive")


@dataclass
class Ghost:
    name: str
    position: np.ndarray
    quaternion: np.ndarray
    drawable: np.ndarray  # (T, bodies); false for unknown mapping/warm-up/world
    position_error: np.ndarray  # versus main FK, NaN where not independent
    angle_error: np.ndarray
    position_tol: float
    angle_tol: float

    @cached_property
    def hotspots(self):
        return (self.position_error > self.position_tol) | (
            self.angle_error > self.angle_tol
        )

    @property
    def failed(self):
        return bool(self.hotspots.any())

    def as_dict(self, names, root):
        entities = {}
        for i, name in enumerate(names):
            if not name:
                continue
            values = {}
            for key, errors, tolerance in (
                ("position", self.position_error, self.position_tol),
                ("angle", self.angle_error, self.angle_tol),
            ):
                valid = np.isfinite(errors[:, i])
                values[key] = {
                    "units": "m" if key == "position" else "rad",
                    "tolerance": tolerance,
                    "verified_frames": int(valid.sum()),
                    "rms": float(np.sqrt(np.mean(errors[valid, i] ** 2)))
                    if valid.any()
                    else None,
                    "peak": float(np.max(errors[valid, i])) if valid.any() else None,
                    "peak_frame": int(np.nanargmax(errors[:, i]))
                    if valid.any()
                    else None,
                }
            verified = np.isfinite(self.position_error[:, i]).any()
            status = (
                "FAIL"
                if self.hotspots[:, i].any()
                else "PASS"
                if verified
                else "UNVERIFIED"
            )
            if i == root and self.name != "body-velocity":
                status = "SHARED_ROOT_INPUT"
            entities[name] = {"status": status, **values}
        return {
            "entities": entities,
            "anomaly_frames": np.flatnonzero(self.hotspots.any(axis=1)).tolist(),
        }


@dataclass
class GhostReport:
    base: ValidationReport
    ghosts: dict[str, Ghost]
    options: GhostOptions
    window_frames: int
    root: int
    joint_bodies: tuple[int, ...]
    present_fields: tuple[str, ...]
    unsupported_fields: tuple[str, ...]

    @property
    def view(self):
        return self.base.view

    @property
    def failed(self):
        return self.base.failed or any(g.failed for g in self.ghosts.values())

    @property
    def complete(self):
        return (
            self.base.complete
            and not self.unsupported_fields
            and len(self.view.data.joint_pos) > self.window_frames
        )

    @property
    def status(self):
        return "FAIL" if self.failed else "PASS" if self.complete else "PARTIAL"

    @cached_property
    def anomaly_frames(self):
        mask = np.zeros(len(self.view.data.joint_pos), dtype=bool)
        mask[self.base.anomaly_frames] = True
        for ghost in self.ghosts.values():
            mask |= ghost.hotspots.any(axis=1)
        return np.flatnonzero(mask)

    def worst(self):
        """Return frame and body, considering numerical AND spatial errors."""
        key, frame, entity = self.base.worst()
        metric = self.base.metrics[key]
        body = (
            self.joint_bodies[entity]
            if metric.names == self.view.joint_names
            else entity
        )
        best = (
            float(np.nan_to_num(metric.error[frame, entity] / metric.tolerance)),
            frame,
            body,
        )
        for ghost in self.ghosts.values():
            score = np.maximum(
                np.nan_to_num(ghost.position_error / ghost.position_tol),
                np.nan_to_num(ghost.angle_error / ghost.angle_tol),
            )
            f, b = np.unravel_index(np.argmax(score), score.shape)
            candidate = (float(score[f, b]), int(f), int(b))
            if candidate[0] > best[0]:
                best = candidate
        return best[1], best[2] or self.root

    def as_dict(self):
        result = self.base.as_dict()
        result.update(
            status=self.status,
            complete=bool(self.complete),
            numerical_status=self.base.status,
            anomaly_frames=self.anomaly_frames.tolist(),
            coverage={
                "present_fields": list(self.present_fields),
                "unsupported_fields": list(self.unsupported_fields),
                "scope": "selected file only; converted-file checks do not certify original source fields",
                "uses": {
                    "joint_pos": "main FK; joint-velocity anchor",
                    "joint_vel": "joint-velocity integration",
                    "body_pos_w": "root pose; pose ghost; body-velocity anchor",
                    "body_quat_w": "root pose; pose ghost; body-velocity anchor",
                    "body_lin_vel_w": "body-velocity translation",
                    "body_ang_vel_w": "body-velocity world rotation",
                    "fps": "playback and integration time",
                    "joint_names": "joint mapping",
                    "body_names": "body mapping, or explicitly selected source layout when absent",
                },
            },
            ghosts={
                "requested_window_s": self.options.window,
                "window_frames": self.window_frames,
                "actual_window_s": self.window_frames / self.view.fps,
                "warmup_policy": "full history required; velocity ghosts hidden before window_frames; never integrate across loops",
                "layers": {
                    name: ghost.as_dict(self.view.body_names, self.root)
                    for name, ghost in self.ghosts.items()
                },
            },
        )
        return result


def build_ghosts(base, model, options=None, present_fields=()):
    options = options or GhostOptions()
    view, source = base.view, base.view.data
    n, bodies = source.body_pos_w.shape[:2]
    frame_window = options.window * view.fps
    if not np.isfinite(frame_window):
        raise ValueError("Ghost window is too large for this frame rate")
    steps = max(1, int(np.floor(frame_window + 0.5)))
    root = model.body("LINK_BASE").id
    ghost_data = {
        "pose": (
            source.body_pos_w.copy(),
            source.body_quat_w.copy(),
            view.pose_comparable,
        ),
        "body-velocity": (
            integrate_linear(
                source.body_pos_w, source.body_lin_vel_w, steps, 1 / view.fps
            ),
            integrate_world_orientation(
                source.body_quat_w, source.body_ang_vel_w, steps, 1 / view.fps
            ),
            view.pose_comparable & view.linear_comparable,
        ),
    }
    joint_pos = integrate_linear(
        source.joint_pos, source.joint_vel, steps, 1 / view.fps
    )
    joint_p = np.full(source.body_pos_w.shape, np.nan)
    joint_q = np.full(source.body_quat_w.shape, np.nan)
    if steps < n:
        fk_input = {
            key: getattr(source, key)[steps:]
            if key.startswith("joint_")
            else getattr(source, key)[steps:, root : root + 1]
            for key in STATE_FIELDS
        }
        fk_input["joint_pos"] = joint_pos[steps:]
        joint_fk = tracking_fk(model, view.joint_names, fk_input)
        joint_p[steps:], joint_q[steps:] = joint_fk.body_pos_w, joint_fk.body_quat_w
    ghost_data["joint-velocity"] = (joint_p, joint_q, np.arange(bodies) != 0)
    ghosts = {}
    for name, (p, q, comparable) in ghost_data.items():
        drawable = np.broadcast_to(comparable, (n, bodies)).copy()
        if name != "pose":
            drawable[:steps] = False
        drawable[:, 0] = False
        if not np.isfinite(p[drawable]).all() or not np.isfinite(q[drawable]).all():
            raise ValueError(f"{name}: reconstruction produced non-finite states")
        error_p = np.linalg.norm(p - base.fk.body_pos_w, axis=-1)
        error_q = orientation_error(q, base.fk.body_quat_w)
        error_p[~drawable] = np.nan
        error_q[~drawable] = np.nan
        if name != "body-velocity":
            error_p[:, root] = error_q[:, root] = np.nan
        ghosts[name] = Ghost(
            name,
            p,
            q,
            drawable,
            error_p,
            error_q,
            base.metrics["body_position_fk"].tolerance
            if name == "pose"
            else options.position_tol,
            base.metrics["body_orientation_fk"].tolerance
            if name == "pose"
            else np.deg2rad(options.angle_tol_deg),
        )
    return GhostReport(
        base,
        ghosts,
        options,
        steps,
        root,
        tuple(int(model.jnt_bodyid[model.joint(name).id]) for name in view.joint_names),
        tuple(sorted(present_fields)),
        tuple(sorted(set(present_fields) - KNOWN_FIELDS)),
    )
