"""T800-specific Manager-Based terms, migrated from unilab.tasks.locomotion.t800."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.envs.mdp import JointPositionAction, JointPositionActionCfg
from unilab.managers import CommandTerm, ManagerTermBase, SceneEntityCfg
from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase
from unilab.tasks.locomotion.g1.manager_terms import feet_phase as _G1FeetPhase
from unilab.tasks.locomotion.g1.manager_terms import (
    feet_phase_contact as _G1FeetPhaseContact,
)
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionCommand,
    MotionCommandCfg,
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)
from unilab.utils.rotation import np_quat_apply_inverse, np_yaw_quat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_LATERAL_FOOT_SENSORS = (
    "left_foot_pos",
    "right_foot_pos",
    "base_link_quaternion",
)
_FOOT_POSITION_SENSORS = ("left_foot_pos", "right_foot_pos")
_FOOT_QUATERNION_SENSORS = ("left_foot_quat", "right_foot_quat")
_FOOT_FORCE_SENSORS = ("force_left_foot", "force_right_foot")
_FOOT_CONTACT_SENSORS = tuple(
    f"{side}_foot_contact_{index}" for side in ("left", "right") for index in range(4)
)


def _finite_real(value: Any, *, label: str, strict_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0 or (strict_positive and result == 0.0):
        relation = "greater than zero" if strict_positive else "non-negative"
        raise ValueError(f"{label} must be {relation}")
    return result


def _pair(value: Any, *, name: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite (min, max) pair")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"{name} minimum {lower} exceeds maximum {upper}")
    return lower, upper


def compute_lateral_feet_penalty(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    base_quat: np.ndarray,
    min_width: float = 0.20,
    sigma: float = 0.04,
) -> np.ndarray:
    """Penalize signed foot separation below a minimum heading-frame width."""
    separation_w = left_foot - right_foot
    separation_heading = np_quat_apply_inverse(np_yaw_quat(base_quat), separation_w)
    signed_width = separation_heading[:, 1]
    deficit = np.maximum(min_width - signed_width, 0.0)
    return 1.0 - np.exp(-np.square(deficit / sigma))


def compute_motion_command_gate(
    command: np.ndarray,
    command_threshold: float = 0.1,
) -> np.ndarray:
    if (
        not isinstance(command, np.ndarray)
        or command.ndim != 2
        or command.shape[1] != 3
    ):
        raise ValueError(
            "motion command must have shape (num_envs, 3), got "
            f"{getattr(command, 'shape', None)}"
        )
    if not np.isfinite(command).all():
        raise ValueError("motion command must contain only finite values")
    threshold = _finite_real(command_threshold, label="command_threshold")
    magnitude = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
    return np.asarray(magnitude > threshold, dtype=np.float32)


def compute_excess_foot_force_penalty(
    foot_forces: np.ndarray,
    body_weight: float,
    force_threshold_bw: float = 1.25,
) -> np.ndarray:
    if not isinstance(foot_forces, np.ndarray) or foot_forces.ndim != 3:
        raise ValueError(
            "foot forces must have shape (num_envs, num_feet, 3), got "
            f"{getattr(foot_forces, 'shape', None)}"
        )
    if foot_forces.shape[1:] != (2, 3):
        raise ValueError(
            f"foot forces must have shape (num_envs, 2, 3), got {foot_forces.shape}"
        )
    weight = _finite_real(body_weight, label="body_weight", strict_positive=True)
    threshold = _finite_real(force_threshold_bw, label="force_threshold_bw")
    force_bw = np.linalg.norm(foot_forces, axis=2) / weight
    excess = np.maximum(force_bw - threshold, 0.0)
    return np.asarray(np.sum(np.square(excess), axis=1), dtype=np.float32)


def compute_stance_feet_orientation_penalty(
    foot_quaternions: np.ndarray,
    foot_contacts: np.ndarray,
    pitch_scale: float = 0.2,
) -> np.ndarray:
    if not isinstance(foot_quaternions, np.ndarray) or foot_quaternions.ndim != 3:
        raise ValueError(
            "foot quaternions must have shape (num_envs, 2, 4), got "
            f"{getattr(foot_quaternions, 'shape', None)}"
        )
    if foot_quaternions.shape[1:] != (2, 4):
        raise ValueError(
            f"foot quaternions must have shape (num_envs, 2, 4), got {foot_quaternions.shape}"
        )
    if (
        not isinstance(foot_contacts, np.ndarray)
        or foot_contacts.shape != foot_quaternions.shape[:2]
    ):
        raise ValueError(
            f"foot contacts must have shape {foot_quaternions.shape[:2]}, "
            f"got {getattr(foot_contacts, 'shape', None)}"
        )
    pitch_weight = _finite_real(pitch_scale, label="pitch_scale")
    roll_component = np.square(foot_quaternions[:, :, 1])
    pitch_component = np.square(foot_quaternions[:, :, 2])
    cost = (roll_component + pitch_weight * pitch_component) * foot_contacts
    return np.asarray(np.sum(cost, axis=1), dtype=np.float32)


def _command(env: ManagerBasedRlEnv, term: str, command_name: str) -> np.ndarray:
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError as exc:
        raise KeyError(f"{term} command '{command_name}' is unavailable") from exc
    if not isinstance(command, np.ndarray) or command.shape != (env.num_envs, 3):
        raise ValueError(
            f"{term} command '{command_name}' must have shape ({env.num_envs}, 3), "
            f"got {getattr(command, 'shape', None)}"
        )
    return command


class command_gated_feet_phase(_G1FeetPhase):
    _allowed_params = _G1FeetPhase._allowed_params | {"command_threshold"}

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._command_threshold = _finite_real(
            cfg.params.get("command_threshold", 0.1),
            label=f"{self.name} command_threshold",
        )
        super().__init__(cfg, env)

    def _gate(self, env: ManagerBasedRlEnv) -> np.ndarray:
        command = _command(env, self.name, self._command_name)
        return compute_motion_command_gate(command, self._command_threshold)


class command_gated_feet_phase_contact(_G1FeetPhaseContact):
    _allowed_params = _G1FeetPhaseContact._allowed_params | {"command_threshold"}

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._command_threshold = _finite_real(
            cfg.params.get("command_threshold", 0.1),
            label=f"{self.name} command_threshold",
        )
        super().__init__(cfg, env)

    def _gate(self, env: ManagerBasedRlEnv) -> np.ndarray:
        command = _command(env, self.name, self._command_name)
        return compute_motion_command_gate(command, self._command_threshold)


class _FootContactTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._contact_view = self._bind(_FOOT_CONTACT_SENSORS)
        if self._contact_view.dimensions != (1,) * 8:
            raise ValueError(
                f"{self.name} contact sensors must expose eight scalars; received "
                f"{self._contact_view.dimensions}"
            )

    def _contacts(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = self._read(self._contact_view, self.name)
        if values.shape != (env.num_envs, 8):
            raise ValueError(
                f"{self.name} contact sensors must have shape ({env.num_envs}, 8), "
                f"got {values.shape}"
            )
        return np.column_stack(
            (np.any(values[:, :4] > 0.5, axis=1), np.any(values[:, 4:] > 0.5, axis=1))
        )


class _FootForceTerm(SensorTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._force_view = self._bind(_FOOT_FORCE_SENSORS)
        if self._force_view.dimensions != (3, 3):
            raise ValueError(
                f"{self.name} force sensors must expose two vec3 values; received "
                f"{self._force_view.dimensions}"
            )

    def _forces(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = self._read(self._force_view, self.name)
        if values.shape != (env.num_envs, 6):
            raise ValueError(
                f"{self.name} force sensors must have shape ({env.num_envs}, 6), "
                f"got {values.shape}"
            )
        return values.reshape(env.num_envs, 2, 3)


class excess_foot_force_l2(_FootForceTerm):
    _allowed_params = frozenset({"body_weight", "force_threshold_bw"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._body_weight = _finite_real(
            cfg.params.get("body_weight"),
            label=f"{self.name} body_weight",
            strict_positive=True,
        )
        self._force_threshold_bw = _finite_real(
            cfg.params.get("force_threshold_bw", 1.25),
            label=f"{self.name} force_threshold_bw",
        )
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return compute_excess_foot_force_penalty(
            self._forces(env),
            self._body_weight,
            self._force_threshold_bw,
        )


class peak_foot_force_bw(_FootForceTerm):
    _allowed_params = frozenset({"body_weight"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._body_weight = _finite_real(
            cfg.params.get("body_weight"),
            label=f"{self.name} body_weight",
            strict_positive=True,
        )
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        force_bw = np.linalg.norm(self._forces(env), axis=2) / self._body_weight
        return np.asarray(np.max(force_bw, axis=1), dtype=np.float32)


class stance_feet_orientation_l2(_FootContactTerm):
    _allowed_params = frozenset({"pitch_scale"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._pitch_scale = _finite_real(
            cfg.params.get("pitch_scale", 0.2), label=f"{self.name} pitch_scale"
        )
        super().__init__(cfg, env)
        self._quaternion_view = self._bind(_FOOT_QUATERNION_SENSORS)
        if self._quaternion_view.dimensions != (4, 4):
            raise ValueError(
                f"{self.name} quaternion sensors must expose two quaternions; received "
                f"{self._quaternion_view.dimensions}"
            )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        quaternions = self._read(self._quaternion_view, self.name)
        if quaternions.shape != (env.num_envs, 8):
            raise ValueError(
                f"{self.name} quaternion sensors must have shape ({env.num_envs}, 8), "
                f"got {quaternions.shape}"
            )
        return compute_stance_feet_orientation_penalty(
            quaternions.reshape(env.num_envs, 2, 4),
            self._contacts(env),
            self._pitch_scale,
        )


class double_stance_metric(_FootContactTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contacts = self._contacts(env)
        return np.asarray(np.all(contacts, axis=1), dtype=np.float32)


class flight_metric(_FootContactTerm):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        contacts = self._contacts(env)
        return np.asarray(np.logical_not(np.any(contacts, axis=1)), dtype=np.float32)


class _FootKinematicsMetric(_FootContactTerm):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._position_view = self._bind(_FOOT_POSITION_SENSORS)
        if self._position_view.dimensions != (3, 3):
            raise ValueError(
                f"{self.name} position sensors must expose two vec3 values; received "
                f"{self._position_view.dimensions}"
            )
        self._step_dt = _finite_real(
            env.step_dt,
            label=f"{self.name} step_dt",
            strict_positive=True,
        )
        self._previous_position = np.zeros((env.num_envs, 2, 3), dtype=np.float32)
        self._previous_contact = np.zeros((env.num_envs, 2), dtype=np.bool_)
        self._initialized = np.zeros(env.num_envs, dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        rows = env_ids if env_ids is not None else slice(None)
        self._previous_position[rows] = 0.0
        self._previous_contact[rows] = False
        self._initialized[rows] = False

    def _sample(
        self, env: ManagerBasedRlEnv
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._read(self._position_view, self.name)
        if values.shape != (env.num_envs, 6):
            raise ValueError(
                f"{self.name} position sensors must have shape ({env.num_envs}, 6), "
                f"got {values.shape}"
            )
        position = values.reshape(env.num_envs, 2, 3)
        contacts = self._contacts(env)
        velocity = (position - self._previous_position) / self._step_dt
        valid = self._initialized.copy()
        self._previous_position[:] = position
        self._previous_contact[:] = contacts
        self._initialized[:] = True
        return velocity, contacts, valid


class touchdown_foot_speed(_FootKinematicsMetric):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        previous_contact = self._previous_contact.copy()
        velocity, contacts, valid = self._sample(env)
        touchdown = contacts & np.logical_not(previous_contact) & valid[:, None]
        downward_speed = np.maximum(-velocity[:, :, 2], 0.0)
        return np.asarray(np.max(downward_speed * touchdown, axis=1), dtype=np.float32)


class stance_foot_slip_speed(_FootKinematicsMetric):
    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        previous_contact = self._previous_contact.copy()
        velocity, contacts, valid = self._sample(env)
        persistent_contact = contacts & previous_contact & valid[:, None]
        planar_speed = np.linalg.norm(velocity[:, :, :2], axis=2)
        return np.asarray(
            np.max(planar_speed * persistent_contact, axis=1), dtype=np.float32
        )


class OfficialMujocoRigidBodyMaterial(ManagerTermBase):
    """Keep one official friction bucket per environment for the whole run.

    UniLab 1.1 exposes geom-friction writes through reset transactions.  The
    original training branch registered the same sampled values as persistent
    startup state.  This downstream term samples once during construction and
    re-applies those cached values on every reset, preserving the trained
    per-environment distribution without reaching into backend-private state.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if getattr(cfg, "mode", None) != "reset":
            raise NotImplementedError(
                "OfficialMujocoRigidBodyMaterial only supports mode='reset'"
            )
        params = cfg.params
        allowed = {
            "asset_cfg",
            "static_friction_range",
            "dynamic_friction_range",
            "restitution_range",
            "num_buckets",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                f"official material randomization has unknown parameters {unknown}"
            )
        ranges = np.asarray(
            (
                _pair(params["static_friction_range"], name="static_friction_range"),
                _pair(params["dynamic_friction_range"], name="dynamic_friction_range"),
                _pair(params["restitution_range"], name="restitution_range"),
            ),
            dtype=np.float64,
        )
        num_buckets = params["num_buckets"]
        if (
            isinstance(num_buckets, bool)
            or not isinstance(num_buckets, int)
            or num_buckets < 1
        ):
            raise ValueError("num_buckets must be a positive integer")
        asset_cfg = params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("asset_cfg must be a SceneEntityCfg")

        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._geom_ids, default_friction = self._entity.bind_geom_friction_write(
            asset_cfg.geom_ids,
            term_name="official_mujoco_rigid_body_material",
        )
        self.material_buckets = env.rng.uniform(
            ranges[:, 0],
            ranges[:, 1],
            size=(num_buckets, 3),
        )
        bucket_ids = env.rng.integers(
            0,
            num_buckets,
            size=(env.num_envs, default_friction.shape[0]),
        )
        self._values = np.broadcast_to(
            default_friction,
            (env.num_envs, *default_friction.shape),
        ).copy()
        # MuJoCo has one sliding coefficient.  Preserve the model's torsional
        # and rolling friction while mapping the official dynamic bucket.
        self._values[:, :, 0] = self.material_buckets[bucket_ids, 1]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice | None,
        asset_cfg: SceneEntityCfg,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
    ) -> None:
        del (
            asset_cfg,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
        )
        ids = (
            np.arange(env.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(env.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self._entity.write_geom_friction_to_sim(
            self._values[ids],
            geom_ids=self._geom_ids,
            env_ids=ids,
            term_name="official_mujoco_rigid_body_material",
        )


class OfficialFixedRigidBodyCom(ManagerTermBase):
    """Sample the official waist CoM offset once and restore it on reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if getattr(cfg, "mode", None) != "reset":
            raise NotImplementedError(
                "OfficialFixedRigidBodyCom only supports mode='reset'"
            )
        params = cfg.params
        if set(params) != {"asset_cfg", "com_range"}:
            raise ValueError(
                "official CoM randomization requires asset_cfg and com_range"
            )
        asset_cfg = params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("asset_cfg must be a SceneEntityCfg")
        com_range = params["com_range"]
        if set(com_range) != {"x", "y", "z"}:
            raise ValueError("com_range must define exactly x, y, and z")
        ranges = np.asarray(
            [_pair(com_range[axis], name=f"com_range.{axis}") for axis in "xyz"],
            dtype=np.float64,
        )
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, default_ipos = self._entity.bind_body_ipos_write(
            asset_cfg.body_ids,
            term_name="official_fixed_rigid_body_com",
        )
        offsets = env.rng.uniform(
            ranges[:, 0],
            ranges[:, 1],
            size=(env.num_envs, 3),
        )
        self._values = default_ipos[None, :, :] + offsets[:, None, :]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice | None,
        asset_cfg: SceneEntityCfg,
        com_range: dict[str, tuple[float, float]],
    ) -> None:
        del asset_cfg, com_range
        ids = (
            np.arange(env.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(env.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self._entity.write_body_ipos_to_sim(
            self._values[ids],
            body_ids=self._body_ids,
            env_ids=ids,
            term_name="official_fixed_rigid_body_com",
        )


@dataclass(kw_only=True)
class OfficialMotionCommandCfg(MotionCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "OfficialMotionCommand":
        return OfficialMotionCommand(self, env)


class OfficialMotionCommand(MotionCommand):
    """Keep the startup encoder bias and trained adaptive-sampler decay."""

    def __init__(self, cfg: OfficialMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        lower, upper = self._joint_default_position_range
        self._startup_joint_default_bias = env.rng.uniform(
            lower,
            upper,
            size=self.joint_default_bias.shape,
        ).astype(self.joint_default_bias.dtype, copy=False)
        self.joint_default_bias[:] = self._startup_joint_default_bias

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(self.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self.joint_default_bias[ids] = self._startup_joint_default_bias[ids]
        return CommandTerm.reset(self, ids)

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        # The trained branch advances this EMA every control step, including
        # steps with no failure.  UniLab 1.1 only advances it on failures.
        if (
            env_ids is None
            and self.sampler.mode == "adaptive"
            and not np.any(self._env.termination_manager.terminated)
        ):
            self.sampler._current_bin_failed.fill(0.0)
            self.sampler.bin_failed_count *= 1.0 - self.sampler.adaptive_alpha
        super()._update_command(env_ids)


@dataclass(kw_only=True)
class OfficialDelayedMotionJointPositionActionCfg(MotionJointPositionActionCfg):
    min_delay: int = 1
    max_delay: int = 3
    delay_groups: dict[str, list[str]]

    def build(
        self, env: ManagerBasedRlEnv
    ) -> "OfficialDelayedMotionJointPositionAction":
        return OfficialDelayedMotionJointPositionAction(self, env)


class OfficialDelayedMotionJointPositionAction(MotionJointPositionAction):
    """Apply one independently sampled physical-step delay per actuator group."""

    requires_substep_state_feedback = True

    def __init__(
        self,
        cfg: OfficialDelayedMotionJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ):
        if cfg.simulate_action_latency:
            raise ValueError(
                "official physical-step delay cannot be combined with action latency"
            )
        if (
            isinstance(cfg.min_delay, bool)
            or isinstance(cfg.max_delay, bool)
            or not isinstance(cfg.min_delay, int)
            or not isinstance(cfg.max_delay, int)
            or cfg.min_delay < 0
            or cfg.min_delay > cfg.max_delay
        ):
            raise ValueError(
                "official action delay requires 0 <= min_delay <= max_delay"
            )
        super().__init__(cfg, env)
        index_by_name = {name: index for index, name in enumerate(self._target_names)}
        groups: list[np.ndarray] = []
        selected: list[str] = []
        for group_name, names in cfg.delay_groups.items():
            if not names:
                raise ValueError(
                    f"delay group '{group_name}' must select at least one joint"
                )
            missing = [name for name in names if name not in index_by_name]
            if missing:
                raise ValueError(
                    f"delay group '{group_name}' has unknown joints {missing}"
                )
            selected.extend(names)
            groups.append(
                np.asarray([index_by_name[name] for name in names], dtype=np.intp)
            )
        duplicates = sorted({name for name in selected if selected.count(name) > 1})
        missing = sorted(set(self._target_names) - set(selected))
        if duplicates or missing:
            raise ValueError(
                "delay_groups must partition every action target once; "
                f"duplicates={duplicates}, missing={missing}"
            )
        self._delay_groups = tuple(groups)
        self._min_delay = cfg.min_delay
        self._max_delay = cfg.max_delay
        self._time_lags = np.zeros((self.num_envs, len(groups)), dtype=np.int32)
        self._history = np.empty(
            (cfg.max_delay + 1, self.num_envs, self.action_dim),
            dtype=self._processed_actions.dtype,
        )
        self._num_pushes = np.zeros(self.num_envs, dtype=np.int32)
        self._pointer = -1
        self._all_env_ids = np.arange(self.num_envs, dtype=np.intp)
        self._delayed_target = np.empty_like(self._target)

    @property
    def target(self) -> np.ndarray:
        return self._delayed_target

    @property
    def time_lags(self) -> np.ndarray:
        return self._time_lags

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.arange(self.num_envs, dtype=np.int32)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self._time_lags[ids] = self._env.rng.integers(
            self._min_delay,
            self._max_delay + 1,
            size=(len(ids), len(self._delay_groups)),
        )
        self._num_pushes[ids] = 0
        self._history[:, ids, :] = 0.0
        self._delayed_target[ids] = 0.0

    def apply_actions(self) -> None:
        entity = cast("Entity", self._entity)
        encoder_bias = entity.data.encoder_bias[:, self._target_ids]
        np.add(
            self._processed_actions,
            self._motion_command.joint_default_bias[:, self._target_ids],
            out=self._target,
        )
        self._target -= encoder_bias

        self._pointer = (self._pointer + 1) % self._history.shape[0]
        self._history[self._pointer] = self._target
        first_push = self._num_pushes == 0
        if np.any(first_push):
            self._history[:, first_push, :] = self._target[first_push][None, :, :]
        self._num_pushes += 1

        for group_index, joint_ids in enumerate(self._delay_groups):
            valid_lags = np.minimum(
                self._time_lags[:, group_index],
                self._num_pushes - 1,
            )
            read_indices = (self._pointer - valid_lags) % self._history.shape[0]
            delayed = self._history[read_indices, self._all_env_ids]
            self._delayed_target[:, joint_ids] = delayed[:, joint_ids]
        entity.set_joint_position_target(
            self._delayed_target,
            joint_ids=self._target_ids,
        )


class official_joint_pos_limits(ManagerTermBase):
    """Official L1 penalty with the soft range materialized on the cold path."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params["asset_cfg"]
        factor = cfg.params["soft_limit_factor"]
        if isinstance(factor, bool) or not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("soft_limit_factor must be finite and in (0, 1]")
        self._asset = cast("Entity", env.scene[asset_cfg.name])
        self._ids = asset_cfg.joint_ids
        limits = self._asset.data.soft_joint_pos_limits[self._ids]
        center = limits.mean(axis=-1)
        half_width = (limits[:, 1] - limits[:, 0]) * (0.5 * factor)
        self._lower = center - half_width
        self._upper = center + half_width

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        soft_limit_factor: float,
    ) -> np.ndarray:
        del env, asset_cfg, soft_limit_factor
        position = self._asset.data.joint_pos[:, self._ids]
        lower = np.maximum(self._lower - position, 0.0)
        upper = np.maximum(position - self._upper, 0.0)
        return np.sum(lower + upper, axis=-1)


class undesired_body_height_contacts(ManagerTermBase):
    """Count selected bodies below the training branch's ground proxy."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("undesired body contacts asset_cfg must be SceneEntityCfg")
        threshold = cfg.params["threshold"]
        if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "undesired body contacts threshold must be finite and non-negative"
            )
        self._asset = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids = asset_cfg.body_ids
        self._threshold = float(threshold)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> np.ndarray:
        del env, threshold, asset_cfg
        positions = self._asset.data.body_link_pos_w[:, self._body_ids, 2]
        return np.sum(positions < self._threshold, axis=-1)


def _validate_unique_selectors(label: str, selectors: Sequence[str]) -> None:
    if len(selectors) != len(set(selectors)):
        raise ValueError(
            f"T800JointPositionAction {label} selectors contain duplicates"
        )


@dataclass(kw_only=True)
class T800JointPositionActionCfg(JointPositionActionCfg):
    """Control selected joints while holding the remaining T800 joints."""

    held_actuator_names: tuple[str, ...] | list[str]

    def build(self, env: ManagerBasedRlEnv) -> "T800JointPositionAction":
        return T800JointPositionAction(self, env)


class T800JointPositionAction(JointPositionAction):
    """Joint-position action with cold-path-resolved non-policy hold targets."""

    def __init__(self, cfg: T800JointPositionActionCfg, env: ManagerBasedRlEnv):
        _validate_unique_selectors("active", cfg.actuator_names)
        _validate_unique_selectors("held", cfg.held_actuator_names)

        super().__init__(cfg, env)
        active_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        held_ids, _ = self._entity.find_joints_by_actuator_names(
            cfg.held_actuator_names
        )
        if len(set(active_ids)) != len(active_ids) or len(set(held_ids)) != len(
            held_ids
        ):
            raise ValueError(
                "T800JointPositionAction resolved joint IDs must be unique"
            )
        if set(active_ids) & set(held_ids):
            raise ValueError("T800JointPositionAction active and held joints overlap")
        if set(active_ids) | set(held_ids) != set(range(self._entity.num_joints)):
            raise ValueError(
                "T800JointPositionAction active and held joints must form a complete partition"
            )

        self._held_ids = np.asarray(held_ids, dtype=np.intp)
        self._held_ids.setflags(write=False)
        held_shape = (self.num_envs, self._held_ids.size)
        self._held_default_targets = np.empty(
            held_shape, dtype=self._entity.data.default_joint_pos.dtype
        )
        self._held_encoder_bias = np.empty(
            held_shape, dtype=self._entity.data.encoder_bias.dtype
        )

    def apply_actions(self) -> None:
        super().apply_actions()
        np.take(
            self._entity.data.default_joint_pos,
            self._held_ids,
            axis=1,
            out=self._held_default_targets,
        )
        np.take(
            self._entity.data.encoder_bias,
            self._held_ids,
            axis=1,
            out=self._held_encoder_bias,
        )
        np.subtract(
            self._held_default_targets,
            self._held_encoder_bias,
            out=self._held_default_targets,
        )
        self._entity.set_joint_position_target(
            self._held_default_targets,
            joint_ids=self._held_ids,
        )


class penalty_close_feet_lateral(SensorTermBase):
    """Penalty for insufficient signed lateral foot separation."""

    _allowed_params = frozenset({"min_width", "sigma"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._min_width = _finite_real(
            cfg.params.get("min_width", 0.20), label=f"{self.name} min_width"
        )
        self._sigma = _finite_real(
            cfg.params.get("sigma", 0.04),
            label=f"{self.name} sigma",
            strict_positive=True,
        )
        self._sensor_view = self._bind(_LATERAL_FOOT_SENSORS)
        if self._sensor_view.dimensions != (3, 3, 4):
            raise ValueError(
                f"{self.name} sensor dimensions must be (3, 3, 4); received "
                f"{self._sensor_view.dimensions} on backend "
                f"'{self._sensor_view.backend_type}'"
            )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = self._read(self._sensor_view, self.name)
        expected_shape = (env.num_envs, 10)
        if values.shape != expected_shape:
            raise ValueError(
                f"{self.name} sensor view must have shape {expected_shape}, got {values.shape}"
            )
        return compute_lateral_feet_penalty(
            values[:, 0:3],
            values[:, 3:6],
            values[:, 6:10],
            min_width=self._min_width,
            sigma=self._sigma,
        )


__all__ = [
    "OfficialDelayedMotionJointPositionAction",
    "OfficialDelayedMotionJointPositionActionCfg",
    "OfficialFixedRigidBodyCom",
    "OfficialMotionCommand",
    "OfficialMotionCommandCfg",
    "OfficialMujocoRigidBodyMaterial",
    "T800JointPositionAction",
    "T800JointPositionActionCfg",
    "command_gated_feet_phase",
    "command_gated_feet_phase_contact",
    "compute_excess_foot_force_penalty",
    "compute_lateral_feet_penalty",
    "compute_motion_command_gate",
    "compute_stance_feet_orientation_penalty",
    "double_stance_metric",
    "excess_foot_force_l2",
    "flight_metric",
    "official_joint_pos_limits",
    "peak_foot_force_bw",
    "penalty_close_feet_lateral",
    "stance_feet_orientation_l2",
    "stance_foot_slip_speed",
    "touchdown_foot_speed",
    "undesired_body_height_contacts",
]
