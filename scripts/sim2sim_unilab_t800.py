#!/usr/bin/env python3
"""Run a UniLab T800 policy on the official SDK MuJoCo model.

The SDK's official model is used directly from ``assets/resource/t800.xml``.
This runner consumes the current UniLab ONNX actor (134 inputs, 25 outputs),
assembles the same observation contract as the SDK dance runner, and applies
the output as a position-PD target.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
from prepare_t800_unilab import check_motion


SDK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = SDK_ROOT / "assets/resource/t800.xml"
DEFAULT_PROFILE = SDK_ROOT / "assets/config/t800/rl_unilab_motion_tracking/default.yaml"
DEFAULT_MOTION = SDK_ROOT / "assets/config/t800/rl_unilab_motion_tracking/motion.npz"
TRAINING_CTRL_DT = 0.02
DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=0"
DEFAULT_LCM_CHANNEL = "virtual_gamepad/gamepad_keys"
GAMEPAD_TIMEOUT = 0.30
GAMEPAD_KEY_NAMES = (
    "LB", "RB", "A", "B", "X", "Y", "BACK", "START",
    "CROSS_X_UP", "CROSS_X_DOWN", "CROSS_Y_LEFT", "CROSS_Y_RIGHT",
)

# Gains from assets/config/t800/pd_stand/default.yaml.
STAND_Q = np.asarray([
    -0.105, 0.105, 0.263, 0.145, -0.051, -0.074,
    -0.105, -0.105, -0.263, 0.145, -0.051, 0.074,
    0.0,
    0.0, 0.05, 0.0, -0.1, 0.0,
    0.0, -0.05, 0.0, -0.1, 0.0,
    0.0, 0.0,
], dtype=np.float32)
STAND_KP = np.asarray([
    1080, 480, 880, 1000, 800, 100, 1080, 480, 880, 1000, 800, 100,
    200, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 100, 100,
], dtype=np.float32)
STAND_KD = np.asarray([
    25, 25, 25, 25, 2, 2, 25, 25, 25, 25, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=np.float32)

# Official SDK passive runner: velocity damping only, no position control.
PASSIVE_KD = np.asarray([
    4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1,
], dtype=np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    out[1:] *= -1.0
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.asarray([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(quat_conj(q), qv), q)[1:]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path quaternion interpolation, returning a unit quaternion."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + float(alpha) * (b - a)
        return (out / max(float(np.linalg.norm(out)), 1e-12)).astype(np.float32)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    w0 = float(np.sin((1.0 - alpha) * theta)) / sin_theta
    w1 = float(np.sin(alpha * theta)) / sin_theta
    return (w0 * a + w1 * b).astype(np.float32)


def upright_yaw_quat(q: np.ndarray) -> np.ndarray:
    """Return an upright quaternion with the input yaw preserved."""
    r = quat_to_mat(q)
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return np.asarray(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32
    )


def id_by_name(model: mujoco.MjModel, kind: int, names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        idx = mujoco.mj_name2id(model, kind, name)
        if idx < 0:
            raise RuntimeError(f"Official SDK model has no {name!r}")
        ids.append(idx)
    return ids


def actuator_ids_for_joints(model: mujoco.MjModel, joint_ids: list[int]) -> np.ndarray:
    """Return the actuator index driving each joint, in policy joint order.

    UniLab calls its actuators by the joint name; the SDK prefixes its torque
    motors with ``motor_``.  Matching the transmission keeps the policy order
    correct without relying on XML actuator ordering.
    """
    result: list[int] = []
    for joint_id in joint_ids:
        matches = np.flatnonzero(
            (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            & (model.actuator_trnid[:, 0] == joint_id)
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one actuator for joint {model.joint(joint_id).name!r}, "
                f"found {len(matches)}"
            )
        result.append(int(matches[0]))
    return np.asarray(result, dtype=np.int32)


def load_motion(path: Path) -> dict[str, np.ndarray]:
    motion = dict(np.load(path, allow_pickle=False))
    fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
    if fps != 50:
        raise ValueError(f"Expected a 50 Hz motion, got fps={fps}")
    if motion["joint_pos"].shape[1] != 25 or motion["joint_vel"].shape[1] != 25:
        raise ValueError("Motion must contain 25 T800 joints")
    motion["body_names"] = np.asarray(motion["body_names"]).astype(str)
    return motion


class VirtualGamepadReceiver:
    """Small receiver for the official SDK virtual-gamepad LCM message."""

    def __init__(self, url: str) -> None:
        import lcm

        message_root = SDK_ROOT / "tools/virtual_gamepad"
        sys.path.insert(0, str(message_root))
        from lcm_msgs.data.GamepadKeys import GamepadKeys

        self._message_type = GamepadKeys
        try:
            self._lcm = lcm.LCM(url)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot open LCM URL {url!r}. Check the multicast route or pass "
                "--lcm-url with the same URL used by the virtual gamepad."
            ) from exc
        self._mask = 0
        self._previous_mask = 0
        self._last_received = 0.0
        self._lcm.subscribe(DEFAULT_LCM_CHANNEL, self._on_message)

    def _on_message(self, _channel: str, payload: bytes) -> None:
        message = self._message_type.decode(payload)
        self._mask = sum((1 << i) for i, value in enumerate(message.digital_states) if value)
        self._last_received = time.monotonic()

    def poll(self) -> tuple[int, int]:
        handled = self._lcm.handle_timeout(0)
        if handled < 0:
            raise RuntimeError("LCM handle_timeout failed while receiving virtual-gamepad input")
        while handled > 0:
            handled = self._lcm.handle_timeout(0)
            if handled < 0:
                raise RuntimeError("LCM handle_timeout failed while receiving virtual-gamepad input")
        current = self._mask if time.monotonic() - self._last_received <= GAMEPAD_TIMEOUT else 0
        rising = current & ~self._previous_mask
        self._previous_mask = current
        return current, rising

    @staticmethod
    def describe(mask: int) -> str:
        names = [name for i, name in enumerate(GAMEPAD_KEY_NAMES) if mask & (1 << i)]
        return "+".join(names) if names else "none"

    def close(self) -> None:
        self._lcm.close()


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: dict[str, np.ndarray],
    frame: int,
    joint_qpos_ids: np.ndarray,
    joint_qvel_ids: np.ndarray,
    root_body_id: int,
    motion_root_id: int,
    last_action: np.ndarray,
    default_q: np.ndarray,
) -> np.ndarray:
    q = data.qpos[joint_qpos_ids].astype(np.float32)
    qd = data.qvel[joint_qvel_ids].astype(np.float32)
    robot_quat = data.xquat[root_body_id].astype(np.float32)  # wxyz in MuJoCo
    ref_quat = motion["body_quat_w"][frame, motion_root_id].astype(np.float32)

    rel_quat = quat_mul(quat_conj(robot_quat), ref_quat)
    # UniLab's helper writes the first two columns in C-order from the 3x3
    # matrix: [R00, R01, R10, R11, R20, R21].
    rel_mat = quat_to_mat(rel_quat)
    anchor_ori_b = rel_mat[:, :2].reshape(6).astype(np.float32)

    # The actor was trained with the named MuJoCo sensors, not with free-joint
    # coordinates.  ``pelvis_local_linvel`` is the velocity at the IMU site
    # (including omega x r), while ``torso_gyro`` is already local-frame
    # angular velocity.  mj_objectVelocity has the same [angular, linear]
    # ordering when flg_local=1.
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if imu_site_id < 0:
        raise RuntimeError("Official SDK model has no 'imu' site")
    # The MuJoCo Python binding requires a float64 result buffer here.
    imu_velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        imu_site_id,
        imu_velocity,
        1,  # local frame
    )
    base_ang_vel_b = imu_velocity[:3].astype(np.float32)
    obs = np.concatenate([
        motion["joint_pos"][frame], motion["joint_vel"][frame],
        anchor_ori_b, base_ang_vel_b,
        q - default_q, qd, last_action,
    ]).astype(np.float32)
    if obs.shape != (134,):
        raise RuntimeError(f"Observation shape is {obs.shape}, expected (134,)")
    return obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--policy", type=Path, required=True,
        help="exported 134->25 ONNX policy; no checkpoint is bundled",
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument(
        "--ctrl-dt", type=float, default=TRAINING_CTRL_DT,
        help="Policy/control period in seconds (checkpoint contract: 0.02)",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Reference frame used for reset and as the first policy input",
    )
    parser.add_argument(
        "--root-z", type=float, default=None,
        help="Override initial root height; default keeps the motion frame height",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Loop motion after the last frame")
    parser.add_argument(
        "--lcm",
        action="store_true",
        help="Start in PD stand mode and use the virtual gamepad to switch modes",
    )
    parser.add_argument(
        "--lcm-url",
        default=DEFAULT_LCM_URL,
        help=f"LCM URL (default: {DEFAULT_LCM_URL})",
    )
    parser.add_argument(
        "--motion-handoff",
        choices=("smooth", "exact"),
        default=None,
        help=(
            "LCM RB+B handoff: smooth keeps the standing state and lets the "
            "policy take over, exact resets to motion frame 0. Default is "
            "smooth with --lcm and exact without it."
        ),
    )
    parser.add_argument(
        "--handoff-ramp-steps",
        type=int,
        default=50,
        help=(
            "For smooth stand/motion handoffs, ramp targets over this many "
            "20 ms control steps (default: 50)."
        ),
    )
    parser.add_argument(
        "--motion-end",
        choices=("stand", "hold", "exit"),
        default=None,
        help=(
            "What to do when a non-looping motion reaches its last frame. "
            "Default is stand with --lcm and hold without it."
        ),
    )
    parser.add_argument(
        "--action-latency",
        action="store_true",
        help="Apply the previous policy action (only for checkpoints trained with latency)",
    )
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    for path, label in ((args.model, "model"), (args.policy, "policy"),
                        (args.motion, "motion"), (args.config, "config")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    profile = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if profile["expected_observation_dim"] != 134 or profile["resident_control"]:
        raise ValueError("sim2sim supports the 134D default-pose action contract only")
    check_motion(args.motion, profile)
    if abs(args.ctrl_dt - 1.0 / profile["expected_motion_fps"]) > 1e-9:
        raise ValueError("control period must match the motion/policy 50 Hz contract")
    joint_names = profile["joint_names"]

    def policy_vector(key: str) -> np.ndarray:
        value = np.asarray(profile[key], dtype=np.float32)
        if value.shape != (25,) or not np.isfinite(value).all():
            raise ValueError(f"invalid {key}: expected 25 finite values")
        return value

    default_q = policy_vector("default_joint_pos")
    action_scale = policy_vector("action_scale")
    kp_motion = policy_vector("joint_stiffness")
    kd_motion = policy_vector("joint_damping")
    target_lower = policy_vector("target_position_lower")
    target_upper = policy_vector("target_position_upper")
    if np.any(target_lower >= target_upper):
        raise ValueError("invalid target position limits")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.handoff_ramp_steps < 0:
        raise ValueError("--handoff-ramp-steps must be non-negative")

    motion_end = args.motion_end or ("stand" if args.lcm else "hold")
    motion_handoff = args.motion_handoff or ("smooth" if args.lcm else "exact")
    handoff_ramp_steps = (
        args.handoff_ramp_steps if args.lcm and motion_handoff == "smooth" else 0
    )

    import onnxruntime as ort

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    if abs(float(model.opt.timestep) - 0.002) > 1e-7:
        print(f"[sim2sim] WARNING: SDK model timestep={model.opt.timestep}; expected 0.002")

    if args.ctrl_dt <= 0.0:
        raise ValueError("--ctrl-dt must be positive")
    substeps_float = args.ctrl_dt / float(model.opt.timestep)
    substeps = int(round(substeps_float))
    if substeps < 1 or abs(substeps_float - substeps) > 1e-6:
        raise ValueError(
            f"ctrl_dt/model timestep must be an integer; got {args.ctrl_dt}/"
            f"{model.opt.timestep}={substeps_float}"
        )
    print(f"[sim2sim] ctrl_dt={args.ctrl_dt:g} substeps={substeps}")

    motion = load_motion(args.motion.resolve())
    joint_ids = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names)
    joint_qpos_ids = np.asarray([model.jnt_qposadr[i] for i in joint_ids], dtype=np.int32)
    joint_qvel_ids = np.asarray([model.jnt_dofadr[i] for i in joint_ids], dtype=np.int32)
    actuator_ids = actuator_ids_for_joints(model, joint_ids)
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")
    motion_names = list(motion["body_names"])
    motion_root_id = int(profile["trajectory_body_index"])
    if motion_names[motion_root_id] != "LINK_BASE":
        raise ValueError("configured root body does not identify LINK_BASE")
    frame_count = motion["joint_pos"].shape[0]
    if args.start_frame < 0 or args.start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}]")

    available_providers = ort.get_available_providers()
    providers = [
        name
        for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if name in available_providers
    ]
    if not providers:
        raise RuntimeError(f"ONNX Runtime has no usable execution provider: {available_providers}")
    session = ort.InferenceSession(str(args.policy.resolve()), providers=providers)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    if inp.shape[-1] != 134 or out.shape[-1] != 25:
        raise RuntimeError(f"Expected ONNX 134->25, got input={inp.shape}, output={out.shape}")
    print(f"[sim2sim] model={args.model.resolve()}")
    print(f"[sim2sim] policy={args.policy.resolve()} providers={session.get_providers()}")
    print(f"[sim2sim] motion={args.motion.resolve()} frames={motion['joint_pos'].shape[0]}")

    if args.lcm:
        # In gamepad mode start from the official T800 stand pose.  The
        # virtual gamepad can then switch to the UniLab motion at runtime.
        # Keep the motion frame's world anchor so the policy does not see a
        # several-metre position/orientation error when motion mode starts.
        root_pos = motion["body_pos_w"][args.start_frame, motion_root_id].astype(np.float64).copy()
        root_quat = motion["body_quat_w"][args.start_frame, motion_root_id].astype(np.float64).copy()
        root_pos[2] = args.root_z if args.root_z is not None else root_pos[2]
        data.qpos[:7] = np.concatenate((root_pos, root_quat))
        data.qpos[joint_qpos_ids] = STAND_Q
        data.qvel[:] = 0.0
        mode = "stand"
    else:
        # Without LCM, start from the selected motion reference frame.
        root_pos = motion["body_pos_w"][args.start_frame, motion_root_id].astype(np.float64).copy()
        root_quat = motion["body_quat_w"][args.start_frame, motion_root_id].astype(np.float64).copy()
        root_pos[2] = args.root_z if args.root_z is not None else root_pos[2]
        data.qpos[:7] = np.concatenate((root_pos, root_quat))
        data.qpos[joint_qpos_ids] = motion["joint_pos"][args.start_frame]
        data.qvel[:] = 0.0
        data.qvel[0:3] = motion["body_lin_vel_w"][args.start_frame, motion_root_id]
        # NPZ root angular velocity is world-frame; MuJoCo free-joint angular
        # velocity is stored in the root body's local frame.
        data.qvel[3:6] = quat_rotate_inverse(
            root_quat.astype(np.float32),
            motion["body_ang_vel_w"][args.start_frame, motion_root_id].astype(np.float32),
        )
        data.qvel[joint_qvel_ids] = motion["joint_vel"][args.start_frame]
        mode = "motion"
    mujoco.mj_forward(model, data)

    q_lower = target_lower
    q_upper = target_upper
    clipped_frames = 0
    clipped_joints = 0
    last_action = np.zeros(25, dtype=np.float32)
    # Keep an opt-in switch for checkpoints whose training contract requires a
    # one-control-step delayed target.  The validated SDK sim2sim path for this
    # checkpoint uses the current action by default; the observation always
    # exposes the current raw policy action, matching ActionManager.last_action.
    applied_action = np.zeros(25, dtype=np.float32)
    gamepad = None
    if args.lcm:
        gamepad = VirtualGamepadReceiver(args.lcm_url)
        print(
            f"[sim2sim] LCM subscribed to {DEFAULT_LCM_CHANNEL} at {args.lcm_url}; "
            f"initial mode=pd_stand, handoff={motion_handoff}, "
            f"ramp_steps={handoff_ramp_steps}"
        )
        print(
            "[sim2sim] keys: RB+B=UniLab dance, LB+A=stand, "
            "LB+RB=passive, LB+B=official walk (SDK runner), BACK=quit; "
            "hold the combo briefly"
        )
    viewer = None
    if not args.headless:
        # ``mujoco.viewer`` is an optional submodule and is not imported by
        # ``import mujoco`` in all supported MuJoCo Python wheels.
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    try:
        step = 0
        stand_step = 0
        handoff_q_start: np.ndarray | None = None
        handoff_ramp_step = handoff_ramp_steps
        stand_transition_q_start: np.ndarray | None = None
        stand_transition_step = 0
        stand_transition_steps = max(1, args.handoff_ramp_steps)
        stand_transition_root_quat_start: np.ndarray | None = None
        stand_transition_root_quat_target: np.ndarray | None = None

        def begin_stand_transition(reason: str) -> None:
            """Leave motion continuously and settle at the SDK stand pose."""
            nonlocal mode, stand_transition_q_start, stand_transition_step
            nonlocal handoff_q_start, handoff_ramp_step
            nonlocal stand_transition_root_quat_start, stand_transition_root_quat_target
            stand_transition_q_start = data.qpos[joint_qpos_ids].astype(np.float32).copy()
            stand_transition_step = 0
            stand_transition_root_quat_start = data.qpos[3:7].astype(np.float32).copy()
            stand_transition_root_quat_target = upright_yaw_quat(stand_transition_root_quat_start)
            handoff_q_start = None
            handoff_ramp_step = handoff_ramp_steps
            # Keep joint qpos continuous at the button edge.  The free base
            # may still be tilted by the dance, so its orientation is moved
            # smoothly to upright (same yaw) during this transition.
            # Damp residual base angular velocity; otherwise the old motion
            # can rotate the base away from the interpolated upright target.
            data.qvel[3:6] = 0.0
            mode = "stand_transition"
            last_action.fill(0.0)
            applied_action.fill(0.0)
            print(
                f"[sim2sim] {reason} -> stand_transition "
                f"({stand_transition_steps} steps)"
            )

        while step < args.steps:
            wall_start = time.perf_counter()
            if gamepad is not None:
                current_mask, rising_mask = gamepad.poll()
                if rising_mask:
                    print(
                        f"[sim2sim] gamepad: {gamepad.describe(current_mask)} "
                        f"(mask=0x{current_mask:03x})"
                    )
                rb_b = (1 << 1) | (1 << 3)
                lb_a = (1 << 0) | (1 << 2)
                lb_rb = (1 << 0) | (1 << 1)
                lb_b = (1 << 0) | (1 << 3)
                start_up = (1 << 7) | (1 << 8)
                start_down = (1 << 7) | (1 << 9)
                if current_mask & rb_b == rb_b and rising_mask & rb_b:
                    q_before_handoff = data.qpos[joint_qpos_ids].copy()
                    q_motion = motion["joint_pos"][args.start_frame]
                    max_handoff_error = float(np.max(np.abs(q_before_handoff - q_motion)))
                    if motion_handoff == "exact":
                        # This is the training reset contract, but it visibly
                        # teleports a large arm pose when entered from stand.
                        data.qpos[:7] = np.concatenate((root_pos, root_quat))
                        data.qpos[joint_qpos_ids] = q_motion
                        data.qvel[:] = 0.0
                        data.qvel[0:3] = motion["body_lin_vel_w"][args.start_frame, motion_root_id]
                        data.qvel[3:6] = quat_rotate_inverse(
                            root_quat.astype(np.float32),
                            motion["body_ang_vel_w"][args.start_frame, motion_root_id].astype(np.float32),
                        )
                        data.qvel[joint_qvel_ids] = motion["joint_vel"][args.start_frame]
                    else:
                        # Keep the standing pose/anchor and let the policy
                        # move toward the first reference over multiple
                        # control cycles.  This avoids the visible qpos and
                        # velocity discontinuity at the button edge.
                        data.qvel[:] = 0.0
                        handoff_q_start = q_before_handoff.astype(np.float32)
                        handoff_ramp_step = 0
                    if motion_handoff == "exact":
                        handoff_q_start = None
                        handoff_ramp_step = handoff_ramp_steps
                    mujoco.mj_forward(model, data)
                    mode = "motion"
                    step = 0
                    last_action.fill(0.0)
                    applied_action.fill(0.0)
                    print(
                        f"[sim2sim] RB+B -> motion tracking ({motion_handoff} handoff, "
                        f"initial max|q-motion|={max_handoff_error:.3f} rad)"
                    )
                elif (current_mask & lb_a) == lb_a and rising_mask & lb_a:
                    begin_stand_transition("LB+A -> pd_stand")
                elif (current_mask & lb_rb) == lb_rb and rising_mask & lb_rb:
                    # Match the official task FSM: LB+RB is passive, not
                    # pd_stand.  Passive mode applies only velocity damping;
                    # it intentionally does not hold the robot upright.
                    mode = "passive"
                    handoff_q_start = None
                    handoff_ramp_step = handoff_ramp_steps
                    stand_transition_q_start = None
                    stand_transition_step = 0
                    last_action.fill(0.0)
                    applied_action.fill(0.0)
                    print("[sim2sim] LB+RB -> passive")
                elif (current_mask & lb_b) == lb_b and rising_mask & lb_b:
                    print(
                        "[sim2sim] LB+B -> official walk requires the SDK "
                        "rl_walking_example_runner; not replaced by UniLab dance"
                    )
                elif (current_mask & start_up) == start_up and rising_mask & start_up:
                    print(
                        "[sim2sim] START+CROSS_X_UP is official supine_to_stance; "
                        "unavailable in this UniLab Python runner (requires the SDK MNN runner)"
                    )
                elif (current_mask & start_down) == start_down and rising_mask & start_down:
                    print(
                        "[sim2sim] START+CROSS_X_DOWN is official stance_to_supine; "
                        "unavailable in this UniLab Python runner (requires the SDK MNN runner)"
                    )
                elif rising_mask & (1 << 6):
                    print("[sim2sim] BACK -> quit")
                    break

            if mode == "stand":
                q_target = STAND_Q
                kp = STAND_KP
                kd = STAND_KD
                frame = -1
            elif mode == "passive":
                # Official passive_runner equivalent: q_des=0, kp=0,
                # configured damping only, so no active standing torque.
                q_target = np.zeros(25, dtype=np.float32)
                kp = np.zeros(25, dtype=np.float32)
                kd = PASSIVE_KD
                frame = -1
            elif mode == "stand_transition":
                alpha = min(
                    1.0,
                    float(stand_transition_step + 1) / float(stand_transition_steps),
                )
                data.qpos[3:7] = quat_slerp(
                    stand_transition_root_quat_start,
                    stand_transition_root_quat_target,
                    alpha,
                )
                q_target = (
                    (1.0 - alpha) * stand_transition_q_start
                    + alpha * STAND_Q
                )
                kp = STAND_KP
                kd = STAND_KD
                frame = -1
            else:
                raw_frame = args.start_frame + step * args.frame_stride
                if not args.loop and raw_frame >= frame_count:
                    if motion_end == "stand":
                        begin_stand_transition("motion finished")
                        continue
                    if motion_end == "exit":
                        print("[sim2sim] motion finished -> exit")
                        break
                frame = raw_frame % frame_count if args.loop else min(raw_frame, frame_count - 1)
                obs = build_observation(
                    model, data, motion, frame, joint_qpos_ids, joint_qvel_ids,
                    root_body_id, motion_root_id, last_action, default_q,
                )
                action = np.asarray(
                    session.run([out.name], {inp.name: obs[None, :]})[0][0],
                    dtype=np.float32,
                )
                if action.shape != (25,) or not np.isfinite(action).all():
                    raise RuntimeError(f"Invalid policy output: shape={action.shape}")
                applied_action = applied_action if args.action_latency else action
                raw_target = default_q + action_scale * applied_action
                q_target = np.clip(raw_target, q_lower, q_upper)
                clipped = np.count_nonzero(raw_target != q_target)
                clipped_frames += int(clipped > 0)
                clipped_joints += int(clipped)
                if handoff_q_start is not None and handoff_ramp_step < handoff_ramp_steps:
                    alpha = float(handoff_ramp_step + 1) / float(handoff_ramp_steps)
                    q_target = (1.0 - alpha) * handoff_q_start + alpha * q_target
                    handoff_ramp_step += 1
                kp = kp_motion
                kd = kd_motion

            for _ in range(substeps):
                # A UniLab <position> actuator evaluates kp*(target-q)-kd*qd
                # at every 2 ms physics step.  The SDK model exposes torque
                # motors, so reproduce that equation at the same cadence.
                q = data.qpos[joint_qpos_ids].astype(np.float32)
                qd = data.qvel[joint_qvel_ids].astype(np.float32)
                torque = kp * (q_target - q) - kd * qd
                # Match the SDK motor ctrlrange (Nm), including saturation.
                torque = np.clip(
                    torque,
                    model.actuator_ctrlrange[actuator_ids, 0],
                    model.actuator_ctrlrange[actuator_ids, 1],
                )
                data.ctrl[actuator_ids] = torque
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if mode == "stand":
                if args.print_every > 0 and stand_step % args.print_every == 0:
                    print(f"[sim2sim] mode=pd_stand step={stand_step:4d} root_z={data.xpos[root_body_id, 2]:.3f}")
                stand_step += 1
            elif mode == "passive":
                if args.print_every > 0 and stand_step % args.print_every == 0:
                    print(f"[sim2sim] mode=passive step={stand_step:4d} root_z={data.xpos[root_body_id, 2]:.3f}")
                stand_step += 1
            elif mode == "stand_transition":
                if args.print_every > 0 and stand_step % args.print_every == 0:
                    alpha = min(
                        1.0,
                        float(stand_transition_step + 1) / float(stand_transition_steps),
                    )
                    print(
                        f"[sim2sim] mode=stand_transition step={stand_step:4d} "
                        f"alpha={alpha:.2f} root_z={data.xpos[root_body_id, 2]:.3f}"
                    )
                stand_transition_step += 1
                stand_step += 1
                if stand_transition_step >= stand_transition_steps:
                    mode = "stand"
                    stand_transition_q_start = None
                    stand_transition_root_quat_start = None
                    stand_transition_root_quat_target = None
                    print("[sim2sim] stand_transition complete -> pd_stand")
            else:
                applied_action = action
                last_action = action
                if args.print_every > 0 and (step % args.print_every == 0 or step == args.steps - 1):
                    print(
                        f"[sim2sim] mode=motion step={step:4d} frame={frame:4d} "
                        f"root_z={data.xpos[root_body_id, 2]:.3f} "
                        f"action_abs_max={np.max(np.abs(action)):.3f}"
                    )
                step += 1
            if args.realtime:
                # Pace the complete control cycle, including ONNX inference,
                # physics stepping, and viewer.sync().  Sleeping only for the
                # nominal physics duration makes the GUI run as fast as the
                # host can render instead of at the intended control rate.
                elapsed = time.perf_counter() - wall_start
                time.sleep(max(0.0, args.ctrl_dt - elapsed))
    finally:
        if viewer is not None:
            viewer.close()
        if gamepad is not None:
            gamepad.close()
    if clipped_frames:
        print(
            f"[sim2sim] WARNING: target-position limits clipped in {clipped_frames} "
            f"policy frames ({clipped_joints} joint targets)"
        )
    print("[sim2sim] completed")


if __name__ == "__main__":
    main()
