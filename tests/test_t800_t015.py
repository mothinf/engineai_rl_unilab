"""T015 configuration and fixed-per-environment DR lifecycle contracts."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root
from unilab.managers import SceneEntityCfg

from engineai_rl_unilab.cli import CONF_ROOT, _build_command, _ensure_registry_env
from engineai_rl_unilab.tasks.t800.manager_terms import (
    OfficialFixedRigidBodyCom,
    OfficialMujocoRigidBodyMaterial,
)

ROOT = Path(__file__).resolve().parents[1]


def owner():
    with initialize_config_dir(
        config_dir=str(package_root() / "conf/flashsac"), version_base="1.3"
    ):
        return compose(
            "config",
            overrides=[
                f"hydra.searchpath=[file://{CONF_ROOT / 'flashsac'}]",
                "task=engineai_t800_t015/mjwarp",
            ],
        )


class RecordingEntity:
    """Capture public backend writes without backend-private state access."""

    def bind_geom_friction_write(self, ids, **kwargs):
        self.friction = np.array([[1.0, 0.02, 0.003], [0.9, 0.01, 0.002]])
        return np.array([2, 4]), self.friction

    def bind_body_ipos_write(self, ids, **kwargs):
        self.ipos = np.array([[0.01, 0.02, 0.03]])
        return np.array([7]), self.ipos

    def write_geom_friction_to_sim(self, values, *, env_ids, **kwargs):
        self.written = (np.asarray(env_ids).copy(), values.copy())

    write_body_ipos_to_sim = write_geom_friction_to_sim


def make_term(kind, seed=19):
    entity = RecordingEntity()
    env = SimpleNamespace(
        num_envs=4, rng=np.random.default_rng(seed), scene={"robot": entity}
    )
    asset = SceneEntityCfg(name="robot")
    if kind == "material":
        params = {
            "asset_cfg": asset,
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        }
        cls = OfficialMujocoRigidBodyMaterial
    else:
        params = {
            "asset_cfg": asset,
            "com_range": {axis: (-0.02, 0.02) for axis in "xyz"},
        }
        cls = OfficialFixedRigidBodyCom
    term = cls(SimpleNamespace(mode="reset", params=params), env)
    return env, entity, term, params


@pytest.mark.parametrize("kind", ["material", "com"])
def test_fixed_samples_are_restored_for_partial_and_full_resets(kind):
    env, entity, term, params = make_term(kind)
    term(env, None, **params)
    original = entity.written[1].copy()
    rng_state = copy.deepcopy(env.rng.bit_generator.state)
    for selection, expected in [
        (np.array([2, 0]), [2, 0]),
        (slice(1, 3), [1, 2]),
        (None, [0, 1, 2, 3]),
    ]:
        term(env, selection, **params)
        np.testing.assert_array_equal(entity.written[0], expected)
        np.testing.assert_array_equal(entity.written[1], original[expected])
    assert env.rng.bit_generator.state == rng_state  # Resets must not draw new samples.
    env2, entity2, term2, params2 = make_term(kind)
    term2(env2, None, **params2)
    np.testing.assert_array_equal(entity2.written[1], original)
    env3, entity3, term3, params3 = make_term(kind, seed=20)
    term3(env3, None, **params3)
    assert not np.array_equal(entity3.written[1], original)
    if kind == "material":
        assert np.all((original[:, :, 0] >= 0.7) & (original[:, :, 0] <= 1.2))
        np.testing.assert_array_equal(
            original[:, :, 1:], np.broadcast_to(entity.friction[:, 1:], (4, 2, 2))
        )
    else:
        assert np.all(np.abs(original - entity.ipos) <= 0.02)


def test_t015_owner_contract():
    cfg = owner()
    assert cfg.training.task_name == "EngineAIT800MotionTracking"
    assert cfg.algo.max_iterations == 25000 and cfg.algo.seed == 1
    assert cfg.env.seed is None  # Original snapshot; trainer resolves algo.seed.
    assert cfg.env.sim_dt == 0.005 and cfg.env.ctrl_dt == 0.02
    assert cfg.algo.algo_params.temp_target_sigma == 0.15
    assert cfg.reward.motion_joint_pos.weight == 4
    assert cfg.reward.motion_joint_pos.params.std == 0.25
    assert cfg.reward.motion_joint_vel.weight == 0
    assert cfg.env.actions.joint_pos.simulate_action_latency
    assert cfg.env.observations.actor.enable_corruption
    assert cfg.env.events.physics_material.func.endswith(
        ".OfficialMujocoRigidBodyMaterial"
    )
    assert cfg.env.events.base_com.func.endswith(".OfficialFixedRigidBodyCom")
    command = _build_command(
        "train",
        ["--algo", "flashsac", "--task", "engineai_t800_t015", "--sim", "mjwarp"],
    )
    assert "task=engineai_t800_t015/mjwarp" in command


def test_t015_cpu_runtime_reset_and_step(monkeypatch):
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner()
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="flashsac"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        obs, _ = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 446)
        material = env.event_manager.get_term_cfg("physics_material").func
        com = env.event_manager.get_term_cfg("base_com").func
        assert isinstance(material, OfficialMujocoRigidBodyMaterial)
        assert isinstance(com, OfficialFixedRigidBodyCom)
        before = (material._values.copy(), com._values.copy())
        for _ in range(6):
            state = env.step(np.zeros((2, 25), dtype=np.float32))
            assert state.obs["obs"].shape == (2, 446)
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(x).all() for x in state.obs.values())
        other_q = env.scene["robot"].data.joint_pos[1].copy()
        env.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(env.scene["robot"].data.joint_pos[1], other_q)
        np.testing.assert_array_equal(material._values, before[0])
        np.testing.assert_array_equal(com._values, before[1])
    finally:
        env.close()
