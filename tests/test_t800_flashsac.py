"""Standalone no-DR FlashSAC task, motion data and backend runtime contracts."""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

from engineai_rl_unilab.cli import CONF_ROOT, _build_command, _ensure_registry_env
from engineai_rl_unilab.tasks.t800.manager_terms import OfficialMotionCommand

ROOT = Path(__file__).resolve().parents[1]
TASK = "engineai_t800_motion_tracking"
IDENTITY = "EngineAIT800MotionTracking"


def owner():
    with initialize_config_dir(
        config_dir=str(package_root() / "conf/flashsac"), version_base="1.3"
    ):
        return compose(
            "config",
            overrides=[
                f"hydra.searchpath=[file://{CONF_ROOT / 'flashsac'}]",
                f"task={TASK}/mjwarp",
            ],
        )


def test_standalone_owner_contract():
    cfg = owner()
    assert cfg.training.task_name == IDENTITY
    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.training.inference_request_timeout_sec == 300
    assert cfg.env.ctrl_dt == 0.02
    assert cfg.env.ctrl_dt / cfg.env.sim_dt == 3
    assert cfg.env.max_episode_seconds / cfg.env.ctrl_dt == 500
    assert cfg.algo.algo == "flashsac"
    assert cfg.algo.algo_log_name == "flash_sac"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 25000
    assert cfg.algo.gamma == 0.99
    assert cfg.algo.tau == 0.05
    assert cfg.algo.updates_per_step == 4
    assert cfg.algo.policy_frequency == 2
    assert cfg.algo.batch_size == 2048
    assert cfg.algo.actor_hidden_dim == 128
    assert cfg.algo.critic_hidden_dim == 256
    assert cfg.algo.num_atoms == 101
    assert cfg.algo.algo_params.normalize_reward
    assert cfg.algo.algo_params.temp_target_sigma == 0.15
    assert cfg.reward.motion_joint_pos.weight == 2.0
    assert cfg.reward.motion_joint_pos.params.std == 0.8
    assert cfg.env.observations.actor.terms.motion_anchor_pos_b is None
    assert cfg.env.observations.actor.terms.base_lin_vel is None
    for group in cfg.env.observations.values():
        assert not group.enable_corruption
        for term in group.terms.values():
            if term is not None:
                assert term.get("history_length", 1) == 1
    assert not cfg.env.actions.joint_pos.simulate_action_latency
    selected = []
    for event in cfg.env.events.values():
        assert event.func == "unilab.envs.mdp.pd_gains"
        assert event.mode == "reset"
        assert event.params.operation == "abs"
        assert event.params.kp_range[0] == event.params.kp_range[1]
        assert event.params.kd_range[0] == event.params.kd_range[1]
        selected.extend(event.params.asset_cfg.actuator_names)
    assert len(selected) == len(set(selected)) == 25
    motion = cfg.env.commands.motion
    play = cfg.play_profile.env.commands.motion
    assert motion._target_.endswith(".OfficialMotionCommandCfg")
    assert play._target_ == motion._target_
    assert motion.params.truncate_on_clip_end
    assert motion.params.sampling_mode == "adaptive"
    assert play.params.sampling_mode == "start"
    assert play.params.motion_file == motion.params.motion_file
    for command in (motion, play):
        for key in ("pose_range", "velocity_range"):
            assert all(list(v) == [0.0, 0.0] for v in command.params[key].values())
        assert list(command.params.joint_position_range) == [0.0, 0.0]
        assert list(command.params.joint_default_position_range) == [0.0, 0.0]
    for mode in ("train", "eval"):
        cmd = _build_command(
            mode, ["--algo", "flashsac", "--task", TASK, "--sim", "mjwarp"]
        )
        assert cmd[1].endswith("train_flashsac.py")
        assert f"task={TASK}/mjwarp" in cmd
        assert ("training.play_only=true" in cmd) == (mode == "eval")


def test_motion_layout_and_action_margin():
    import mujoco

    cfg = owner()
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg.env.scene.model_file))
    names = list(cfg.env.scene.entities.robot.joint_names)
    assert names == [model.joint(i).name for i in range(1, model.njnt)]
    scales = []
    for name in names:
        matches = [
            value
            for pattern, value in cfg.env.actions.joint_pos.scale.items()
            if re.fullmatch(pattern, name)
        ]
        assert len(matches) == 1
        scales.append(matches[0])
    np.testing.assert_array_equal(scales, [2.5] * 13 + [3.0] * 10 + [0.2] * 2)
    with np.load(ROOT / cfg.env.commands.motion.params.motion_file) as motion:
        assert motion["fps"].item() == 1.0 / cfg.env.ctrl_dt
        assert motion["joint_names"].tolist() == names
        assert motion["body_names"][0] == ""  # World-body placeholder.
        assert motion["body_names"][1:].tolist() == [
            model.body(i).name for i in range(1, model.nbody)
        ]
        assert motion["joint_pos"].shape == (900, 25)
        for value in motion.values():
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all()
        stand = model.key_qpos[model.key("stand").id, 7:]
        assert np.all(
            np.max(np.abs(motion["joint_pos"] - stand), axis=0) * 1.1 <= scales
        )
        data = mujoco.MjData(model)
        for frame in (0, 450, 899):
            data.qpos[:3] = motion["body_pos_w"][frame, 1]
            data.qpos[3:7] = motion["body_quat_w"][frame, 1]
            data.qpos[7:] = motion["joint_pos"][frame]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(
                data.xpos, motion["body_pos_w"][frame], atol=1e-5
            )


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
def test_runtime_reset_step_and_sampler_decay(monkeypatch, backend):
    if backend == "mjwarp" and os.environ.get("T800_TEST_MJWARP") != "1":
        pytest.skip("Set T800_TEST_MJWARP=1 for the CUDA runtime smoke test")
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner()
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="flashsac"
    ).build_task_env_cfg_override()
    env = registry.make(
        IDENTITY, num_envs=2, sim_backend=backend, env_cfg_override=override
    )
    try:
        env.init_state()
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict) and isinstance(info, dict)
        assert env.obs_groups_spec == {"obs": 134, "critic": 278}
        assert env.action_space.shape == (25,)
        command = env.command_manager.get_term("motion")
        assert isinstance(command, OfficialMotionCommand)
        np.testing.assert_array_equal(command.joint_default_bias, 0.0)
        command.sampler.bin_failed_count.fill(1.0)
        env.termination_manager.terminated.fill(False)
        command._update_command(None)
        np.testing.assert_allclose(
            command.sampler.bin_failed_count, 1.0 - command.sampler.adaptive_alpha
        )
        for _ in range(6):
            state = env.step(np.zeros((2, 25), dtype=np.float32))
            assert state.obs["obs"].shape == (2, 134)
            assert state.obs["critic"].shape == (2, 278)
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(value).all() for value in state.obs.values())
        other = env.scene["robot"].data.joint_pos[1].copy()
        env.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(env.scene["robot"].data.joint_pos[1], other)
        np.testing.assert_array_equal(command.joint_default_bias, 0.0)
    finally:
        env.close()
