"""T800 NPZ schema and nominal MuJoCo kinematics shared by offline tools.

The explicit t800_isaac_v1 layout describes the verified EngineAI export, not
arbitrary IsaacLab files. Root velocities are world-frame body-origin velocities;
non-root source COM/frame relationships are not assumed. No training hooks.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from unilab.tasks.motion_tracking.common.motion_loader import MotionData
from unilab.utils.rotation import np_quat_apply_inverse


SOURCE_BODY_NAMES = (
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_PITCH_R",
    "LINK_WAIST_YAW",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_HEAD_PITCH",
    "LINK_HIP_YAW_L",
    "LINK_HIP_YAW_R",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_HEAD_YAW",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
    "LINK_SHOULDER_YAW_L",
    "LINK_SHOULDER_YAW_R",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_PITCH_R",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_PITCH_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
    "LINK_FOOT_L",
    "LINK_FOOT_R",
    "LINK_WRIST_END_L",
    "LINK_WRIST_END_R",
)
STATE_FIELDS = tuple(MotionData.__dataclass_fields__)


def _names(array: np.ndarray, label: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{label} must be a one-dimensional string array")
    names = tuple(array.astype(str).tolist())
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names")
    return names


def read_motion_npz(
    path: str | Path, *, num_bodies: int
) -> tuple[int, tuple[str, ...], tuple[str, ...] | None, dict[str, np.ndarray]]:
    """Shared structural checks; callers must additionally enforce frame semantics."""
    with np.load(path, allow_pickle=False) as source:
        missing = {"fps", "joint_names", *STATE_FIELDS}.difference(source.files)
        if missing:
            raise ValueError(f"{path}: missing NPZ fields {sorted(missing)}")
        fps_array = np.asarray(source["fps"])
        if fps_array.size != 1 or fps_array.dtype.kind not in "fiu":
            raise ValueError("fps must contain one positive integer frame rate")
        fps = float(fps_array.reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0 or not fps.is_integer():
            raise ValueError("fps must be a positive integer frame rate")
        names = _names(source["joint_names"], "joint_names")
        if len(names) != 25:
            raise ValueError("T800 motions require 25 named joints")
        body_names = (
            _names(source["body_names"], "body_names")
            if "body_names" in source.files
            else None
        )
        if body_names is not None and len(body_names) != num_bodies:
            raise ValueError(f"body_names must contain {num_bodies} names")
        arrays = {}
        for field in STATE_FIELDS:
            value = source[field]
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{field} must contain finite numeric values")
            arrays[field] = value.astype(np.float64)

    positions = arrays["joint_pos"]
    if positions.ndim != 2 or positions.shape[0] == 0:
        raise ValueError("joint_pos must be a nonempty (frames, 25) array")
    frames = positions.shape[0]
    for field, value in arrays.items():
        expected = (
            (frames, 25)
            if field.startswith("joint_")
            else (frames, num_bodies, 4 if field == "body_quat_w" else 3)
        )
        if value.shape != expected:
            raise ValueError(
                f"{field}: expected {expected}, got {value.shape}; "
                "check the explicitly selected source format/body layout"
            )
    norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        raise ValueError("body_quat_w must contain unit wxyz quaternions")
    return int(fps), names, body_names, arrays


def load_t800_isaac_source(
    path: str | Path,
) -> tuple[int, tuple[str, ...], dict[str, np.ndarray]]:
    """Read the source without regenerating or discarding its body states."""
    fps, names, bodies, arrays = read_motion_npz(path, num_bodies=30)
    if bodies is not None and bodies != SOURCE_BODY_NAMES:
        raise ValueError("body_names do not match t800_isaac_v1 source order")
    return fps, names, arrays


def tracking_fk(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    source: dict[str, np.ndarray],
) -> MotionData:
    """Use an independent nominal model; never touch a running simulation."""
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")
    free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (
        root_id < 1
        or len(free_ids) != 1
        or model.jnt_bodyid[free_ids[0]] != root_id
        or model.body_parentid[root_id] != 0
    ):
        raise ValueError("target must have exactly one free root named LINK_BASE")
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    if any(
        j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids
    ):
        raise ValueError("target joint_names must identify scalar hinge joints")
    if model.njnt != len(joint_ids) + 1:
        raise ValueError("target contains joints outside the T800 motion contract")
    qadr = model.jnt_qposadr[joint_ids]
    vadr = model.jnt_dofadr[joint_ids]
    root_qadr = int(model.jnt_qposadr[free_ids[0]])
    root_vadr = int(model.jnt_dofadr[free_ids[0]])
    frames = source["joint_pos"].shape[0]
    result = MotionData(
        joint_pos=source["joint_pos"].astype(np.float32),
        joint_vel=source["joint_vel"].astype(np.float32),
        body_pos_w=np.empty((frames, model.nbody, 3), dtype=np.float32),
        body_quat_w=np.empty((frames, model.nbody, 4), dtype=np.float32),
        body_lin_vel_w=np.empty((frames, model.nbody, 3), dtype=np.float32),
        body_ang_vel_w=np.empty((frames, model.nbody, 3), dtype=np.float32),
    )
    root_quat = source["body_quat_w"][:, 0].copy()
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    root_omega_local = np_quat_apply_inverse(root_quat, source["body_ang_vel_w"][:, 0])
    data = mujoco.MjData(model)
    velocity = np.empty(6, dtype=np.float64)
    for frame in range(frames):
        data.qpos[root_qadr : root_qadr + 3] = source["body_pos_w"][frame, 0]
        data.qpos[root_qadr + 3 : root_qadr + 7] = root_quat[frame]
        data.qvel[root_vadr : root_vadr + 3] = source["body_lin_vel_w"][frame, 0]
        data.qvel[root_vadr + 3 : root_vadr + 6] = root_omega_local[frame]
        data.qpos[qadr] = source["joint_pos"][frame]
        data.qvel[vadr] = source["joint_vel"][frame]
        mujoco.mj_forward(model, data)
        result.body_pos_w[frame] = data.xpos
        result.body_quat_w[frame] = data.xquat
        for body_id in range(model.nbody):
            # XBODY selects the link frame origin, not the inertial/COM frame.
            # flg_local=0 expresses both parts in world axes; order is rot:lin.
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 0
            )
            result.body_ang_vel_w[frame, body_id] = velocity[:3]
            result.body_lin_vel_w[frame, body_id] = velocity[3:]
    if any(not np.isfinite(getattr(result, field)).all() for field in STATE_FIELDS):
        raise ValueError("FK produced non-finite reference data")
    return result
