"""External owner, motion layout, and runtime contract checks."""

from pathlib import Path

import numpy as np
import pytest
from engineai_rl_unilab.cli import CONF_ROOT, _build_command, _ensure_registry_env
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

ROOT = Path(__file__).resolve().parents[1]
TASK = "engineai_t800_motion_tracking"
IDENTITY = "EngineAIT800MotionTracking"


def owner():
    with initialize_config_dir(
        config_dir=str(package_root() / "conf/ppo"), version_base="1.3"
    ):
        return compose(
            "config",
            overrides=[
                f"hydra.searchpath=[file://{CONF_ROOT / 'ppo'}]",
                f"task={TASK}/mujoco",
            ],
        )


def test_mujoco_owner_contract():
    cfg = owner()
    assert cfg.training.task_name == IDENTITY
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.max_episode_seconds / cfg.env.ctrl_dt == 500
    assert cfg.env.ctrl_dt / cfg.env.sim_dt == 3
    assert cfg.algo.max_iterations == 40000
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.seed == 1
    assert cfg.algo.save_interval == 500
    assert cfg.env.commands.motion.params.motion_file.endswith("first18s.npz")
    assert cfg.env.commands.motion.motion_adapter == "t800_isaac_v1"
    assert cfg.env.commands.motion.motion_model_file == cfg.env.scene.model_file
    assert cfg.env.observations.actor.terms.motion_anchor_pos_b is None
    assert cfg.env.observations.actor.terms.base_lin_vel is None
    assert cfg.env.observations.actor.enable_corruption
    assert cfg.env.observations.critic.terms.motion_anchor_pos_b is not None
    assert cfg.env.observations.critic.terms.base_lin_vel is not None
    assert not cfg.env.actions.joint_pos.simulate_action_latency
    assert (
        cfg.env.actions.joint_pos.min_delay,
        cfg.env.actions.joint_pos.max_delay,
    ) == (1, 3)
    delayed_joints = [
        joint
        for group in cfg.env.actions.joint_pos.delay_groups.values()
        for joint in group
    ]
    assert len(delayed_joints) == len(set(delayed_joints)) == 25
    assert "deadzone" not in cfg.env.actions.joint_pos
    assert cfg.reward.action_rate_l2.weight == -0.075
    assert cfg.reward.undesired_contacts.weight == -0.1
    assert cfg.reward.undesired_contacts.params.threshold == 0.05
    assert (
        cfg.reward.joint_limit.func
        == "engineai_rl_unilab.tasks.t800.manager_terms.official_joint_pos_limits"
    )
    assert cfg.reward.joint_limit.params.soft_limit_factor == 0.9
    material, com, push = (
        cfg.env.events.physics_material,
        cfg.env.events.base_com,
        cfg.env.events.push_robot,
    )
    assert material.mode == com.mode == "reset"
    assert material.params.num_buckets == 64
    assert list(material.params.dynamic_friction_range) == [0.3, 1.2]
    assert all(list(com.params.com_range[axis]) == [-0.1, 0.1] for axis in "xyz")
    assert list(push.interval_range_s) == [1.0, 3.0]
    assert list(cfg.env.observations.critic.terms) == [
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "body_pos",
        "body_ori",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    for term in cfg.env.observations.actor.terms.values():
        if term is None:
            continue
        assert term.get("delay_max_lag", 0) == 0
    assert all(
        "noise" not in term for term in cfg.env.observations.critic.terms.values()
    )
    for mode in ("train", "eval"):
        command = _build_command(
            mode, ["--algo", "ppo", "--task", TASK, "--sim", "mujoco"]
        )
        assert f"task={TASK}/mujoco" in command


def test_motion_matches_robot_body_and_joint_order():
    import mujoco
    from engineai_rl_unilab.tasks.t800.motion_adapter import adapt_t800_isaac_motion

    cfg = owner()
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg.env.scene.model_file))
    joint_names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    adapted = adapt_t800_isaac_motion(
        ROOT / cfg.env.commands.motion.params.motion_file,
        model_file=ROOT / cfg.env.scene.model_file,
        joint_names=joint_names,
    )
    assert adapted.joint_names == joint_names
    assert adapted.body_names == (
        "",
        *(model.body(i).name for i in range(1, model.nbody)),
    )
    motion = adapted.data
    data = mujoco.MjData(model)
    for frame in (0, len(motion.joint_pos) // 2, len(motion.joint_pos) - 1):
        data.qpos[:3] = motion.body_pos_w[frame, 1]
        data.qpos[3:7] = motion.body_quat_w[frame, 1]
        data.qpos[7:] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos, motion.body_pos_w[frame], atol=1e-5)
        quat_dot = np.abs(np.sum(data.xquat * motion.body_quat_w[frame], axis=-1))
        np.testing.assert_allclose(quat_dot, 1.0, atol=1e-5)


def test_v2_reference_repairs_foot_velocity_frame_without_changing_pose():
    motion_dir = ROOT / "assets/motions/t800"
    with (
        np.load(motion_dir / "dance1_subject2_t800_first18s_mujoco.npz") as old,
        np.load(motion_dir / "dance1_subject2_t800_first18s_mujoco_v2.npz") as fixed,
    ):
        for key in (
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
        ):
            np.testing.assert_array_equal(fixed[key], old[key])
        names = list(fixed["body_names"])
        fps = float(np.asarray(fixed["fps"]).reshape(-1)[0])
        velocity_fd = np.gradient(
            fixed["body_pos_w"].astype(np.float64), 1 / fps, axis=0
        )
        for name in ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"):
            body_id = names.index(name)
            old_rms = np.sqrt(
                np.mean(
                    (old["body_lin_vel_w"][:, body_id] - velocity_fd[:, body_id]) ** 2
                )
            )
            fixed_rms = np.sqrt(
                np.mean(
                    (fixed["body_lin_vel_w"][:, body_id] - velocity_fd[:, body_id]) ** 2
                )
            )
            assert fixed_rms < 0.03
            assert fixed_rms < old_rms / 10


@pytest.mark.parametrize("adapt_source", [True, False])
def test_runtime_reset_step_and_partial_reset(monkeypatch, adapt_source):
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner()
    if not adapt_source:
        # Missing opt-in fields must continue to materialize the original loader.
        del cfg.env.commands.motion.motion_adapter
        del cfg.env.commands.motion.motion_model_file
        cfg.env.commands.motion.params.motion_file = (
            "assets/motions/t800/dance1_subject2_t800_first18s_mujoco_v2.npz"
        )
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        IDENTITY, num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.init_state()
        from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader
        from engineai_rl_unilab.tasks.t800.motion_loader import T800MotionLoader

        motion = env.command_manager.get_term("motion").motion
        assert type(motion) is (T800MotionLoader if adapt_source else MotionLoader)
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict) and isinstance(info, dict)
        assert env.obs_groups_spec == {"obs": 134, "critic": 275}
        action = env.action_manager.get_term("joint_pos")
        assert action.time_lags.shape == (2, 7)
        assert np.all((action.time_lags >= 1) & (action.time_lags <= 3))
        pushes = action._num_pushes.copy()
        state = env.step(np.zeros((2, 25), dtype=np.float32))
        np.testing.assert_array_equal(action._num_pushes, pushes + 3)
        for _ in range(159):
            state = env.step(np.zeros((2, 25), dtype=np.float32))
            assert state.obs["obs"].shape == (2, 134)
            assert state.obs["critic"].shape == (2, 275)
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(value).all() for value in state.obs.values())
        material = env.event_manager.get_term_cfg("physics_material").func
        base_com = env.event_manager.get_term_cfg("base_com").func
        assert material.material_buckets.shape == (64, 3)
        material_values = material._values.copy()
        com_values = base_com._values.copy()
        raw = np.full((2, 25), 0.001, dtype=np.float32)
        original = raw.copy()
        action.process_actions(raw)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(action.raw_action, raw)
        action.process_actions(np.zeros_like(raw))
        before = action.processed_action.copy()
        raw[:] = 0.1
        action.process_actions(raw)
        np.testing.assert_allclose(
            action.processed_action - before, raw * action.scale, atol=1e-7
        )
        action.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(action.raw_action[0], 0.0)
        np.testing.assert_array_equal(action.raw_action[1], raw[1])
        other_joints = env.scene["robot"].data.joint_pos[1].copy()
        env.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(material._values, material_values)
        np.testing.assert_array_equal(base_com._values, com_values)
        np.testing.assert_array_equal(
            env.scene["robot"].data.joint_pos[1], other_joints
        )
        state = env.step(np.zeros_like(raw))
        assert all(np.isfinite(value).all() for value in state.obs.values())
    finally:
        env.close()


def test_official_l1_joint_limit_penalty():
    from types import SimpleNamespace

    from engineai_rl_unilab.tasks.t800.manager_terms import official_joint_pos_limits

    positions = np.array([[-1.2, 8.0, 1.3], [-0.95, -8.0, 0.95], [0.0, 8.0, 0.0]])
    data = SimpleNamespace(
        joint_pos=positions, soft_joint_pos_limits=np.array([[-1.0, 1.0]] * 3)
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    selection = SimpleNamespace(name="robot", joint_ids=np.array([2, 0]))
    term = official_joint_pos_limits(
        SimpleNamespace(params={"asset_cfg": selection, "soft_limit_factor": 0.9}), env
    )
    np.testing.assert_allclose(term(None, selection, 0.9), [0.7, 0.1, 0.0])


def test_floor_allows_robot_material_bucket_to_own_contact_friction():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/robots/t800/scene_flat.xml")
    )
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    sampled_sliding_friction = 0.37

    assert model.geom_friction[floor_id, 0] == 0.0
    model.geom_friction[1:, 0] = sampled_sliding_friction
    data.qpos[:] = model.key_qpos[key_id]
    data.qpos[2] -= 0.01
    mujoco.mj_forward(model, data)
    floor_contacts = [
        contact
        for contact in data.contact
        if floor_id in (contact.geom1, contact.geom2)
    ]
    assert floor_contacts
    assert all(
        np.isclose(contact.friction[0], sampled_sliding_friction)
        for contact in floor_contacts
    )
