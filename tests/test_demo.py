"""Paired assets, training contracts and deterministic presentation profiles."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from engineai_rl_unilab import demo
from engineai_rl_unilab.cli import _build_command, _ensure_registry_env
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.utils.sim2sim import CrossBackendIncompatibleError, resolve_sim2sim_config

from test_t800_motion_tracking import IDENTITY, ROOT, owner


def test_baseline_and_dr_are_separate():
    baseline, dr = owner(None), owner("wostate_dr")
    assert baseline.algo.max_iterations == 15000
    assert dr.algo.max_iterations == 40000
    assert not baseline.env.events
    assert dr.env.events.physics_material is not None
    assert baseline.env.observations.actor.terms.base_lin_vel is not None
    assert dr.env.observations.actor.terms.base_lin_vel is None
    assert baseline.env.commands.motion.params.motion_file.endswith("_v2.npz")
    command = _build_command(
        "train",
        [
            "--algo",
            "ppo",
            "--task",
            "engineai_t800_motion_tracking",
            "--sim",
            "mujoco",
            "--profile",
            "wostate_dr",
        ],
    )
    assert "task=engineai_t800_motion_tracking/mujoco_wostate_dr" in command


def test_default_baseline_runtime(monkeypatch):
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner(None)
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        IDENTITY, num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.init_state()
        env.reset(np.array([0], dtype=np.int32))
        assert env.obs_groups_spec == {"obs": 140, "critic": 275}
        assert not env.action_manager.get_term("joint_pos").cfg.simulate_action_latency
        state = env.step(np.zeros((1, 25), dtype=np.float32))
        assert all(np.isfinite(v).all() for v in state.obs.values())
    finally:
        env.close()


@pytest.mark.parametrize("name", demo.DEMO_REGISTRY)
@pytest.mark.parametrize("sim", ["mujoco", "motrix"])
def test_paired_checkpoint_contract_and_profile(name, sim):
    spec = demo.DEMO_REGISTRY[name]
    cfg = owner(spec.profile, sim)
    cfg.training.play_only = True
    assert cfg.training.sim_backend == sim
    if sim == "motrix":
        assert cfg.training.play_env_num == 16
        assert cfg.training.render_spacing == 2.0
        assert cfg.training.play_render_mode == "interactive"
        assert cfg.training.play_steps is None
        if name == "t800-dance-18s":
            assert cfg.env.motrix_max_iterations == 10
    directory = demo.ASSETS_ROOT / "checkpoints/t800" / spec.asset_name
    resolve_sim2sim_config(directory, cfg, algo_name="ppo")
    original = json.loads((directory / "run_config.json").read_text())
    assert cfg.env.sim_dt == original["config"]["env"]["sim_dt"]
    assert cfg.env.ctrl_dt == original["config"]["env"]["ctrl_dt"]
    checkpoint = torch.load(
        directory / "model.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["actor_state_dict"]["mlp.0.weight"].shape == (512, 140)
    assert checkpoint["actor_state_dict"]["mlp.6.weight"].shape == (25, 128)
    assert checkpoint["critic_state_dict"]["mlp.0.weight"].shape == (512, 275)
    _ensure_registry_env()
    registry.ensure_registries()
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_play_env_cfg_override()
    params = override["commands"]["motion"]["params"]
    assert params["sampling_mode"] == "start"
    assert params["truncate_on_clip_end"]
    assert all(v == [0, 0] for v in params["pose_range"].values())
    assert override["terminations"]["motion_clip_end"]["time_out"]
    if name == "t800-dance-18s":
        # Historical zero deadzone is exactly identity; the current action
        # constructor has no such argument. Never strip a nonzero deadzone.
        assert cfg.env.actions.joint_pos.deadzone == 0
        expected = OmegaConf.to_container(cfg.env.actions, resolve=True)
        del expected["joint_pos"]["deadzone"]
        assert override["actions"] == expected
        assert cfg.env.commands.motion.params.joint_position_range == [0, 0]
    else:
        assert cfg.env.commands.motion.params.joint_position_range == [-0.1, 0.1]
    cfg.env.observations.actor.terms.base_lin_vel = None
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(directory, cfg, algo_name="ppo")


@pytest.mark.parametrize("name", demo.DEMO_REGISTRY)
@pytest.mark.parametrize("sim", ["mujoco", "motrix"])
def test_demo_runtime_reset_step_and_clip_end(name, sim, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    spec = demo.DEMO_REGISTRY[name]
    cfg = owner(spec.profile, sim)
    cfg.training.play_only = True
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_play_env_cfg_override()
    env = registry.make(
        IDENTITY, num_envs=1, sim_backend=sim, env_cfg_override=override
    )
    try:
        env.init_state()
        obs, _ = env.reset(np.array([0], dtype=np.int32))
        assert env.obs_groups_spec == {"obs": 140, "critic": 275}
        assert np.isfinite(obs["obs"]).all()
        motion = env.command_manager.get_term("motion")
        assert motion.time_steps[0] == 0
        state = env.step(np.zeros((1, 25), dtype=np.float32))
        assert all(np.isfinite(v).all() for v in state.obs.values())
        # Drive the real termination path at the final frame, not a manual
        # frame wrap that leaves the simulated robot at its final pose.
        motion.sampler.current_frames[:] = motion.sampler.current_clip_end_frames
        env.step(np.zeros((1, 25), dtype=np.float32))
        assert motion.time_steps[0] == 0
    finally:
        env.close()
    if name == "t800-dance":
        assert "WARNING" in capsys.readouterr().err


def test_demo_cli_resources_and_subprocess(monkeypatch, tmp_path):
    command = demo.build_demo_command("t800-dance-18s", steps=12, render_mode="record")
    assert "training.play_steps=12" in command
    assert "training.play_render_mode=record" in command
    assert "training.play_only=true" in command
    assert "task=engineai_t800_motion_tracking/motrix_demo18s" in command
    live = demo.build_demo_command("t800-dance-18s")
    assert "training.play_render_mode=interactive" in live
    assert "training.play_steps=null" in live
    with pytest.raises(ValueError, match="record only"):
        demo.build_demo_command("t800-dance-18s", sim="mujoco")
    with pytest.raises(ValueError, match="only to"):
        demo.build_demo_command("t800-dance-18s", steps=10)
    mujoco_record = demo.build_demo_command(
        "t800-dance-18s", sim="mujoco", render_mode="record"
    )
    assert "task=engineai_t800_motion_tracking/mujoco_demo18s" in mujoco_record
    assert "training.play_steps=900" in mujoco_record
    with pytest.raises(FileNotFoundError, match="run_config.json"):
        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()
        demo.build_demo_command("t800-dance-18s", checkpoint=checkpoint)
    (tmp_path / "run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="contract_snapshot"):
        demo.build_demo_command("t800-dance-18s", checkpoint=checkpoint)
    calls = []
    monkeypatch.setattr(
        demo.subprocess,
        "run",
        lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=17),
    )
    monkeypatch.chdir(tmp_path)
    assert demo.main(["t800-dance", "--render-mode", "record", "--steps", "5"]) == 17
    assert calls[0][1]["cwd"] == ROOT
    for args in (
        ["unknown"],
        ["t800-dance-18s", "--steps", "0"],
        ["t800-dance-18s", "--render-mode", "none"],
    ):
        with pytest.raises(SystemExit) as exc:
            demo.main(args)
        assert exc.value.code == 2


def test_unified_entry_preserves_unilab_demos(monkeypatch):
    from unilab import cli as upstream_cli

    calls = []
    monkeypatch.setattr(
        upstream_cli, "demo_main", lambda argv: calls.append(argv) or 23
    )
    assert demo.main(["dance", "--device", "cpu"]) == 23
    assert calls == [["dance", "--device", "cpu"]]


def test_resource_hashes():
    manifest = json.loads(
        (demo.ASSETS_ROOT / "checkpoints/t800/manifest.json").read_text()
    )
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("name", demo.DEMO_REGISTRY)
@pytest.mark.parametrize("sim", ["mujoco", "motrix"])
def test_policy_completes_two_full_clips(name, sim, monkeypatch):
    """Use the shipped learned policy, not zero actions or pose overwrites."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    spec = demo.DEMO_REGISTRY[name]
    cfg = owner(spec.profile, sim)
    cfg.training.play_only = True
    num_envs = int(cfg.training.play_env_num)
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_play_env_cfg_override()
    env = registry.make(
        IDENTITY, num_envs=num_envs, sim_backend=sim, env_cfg_override=override
    )
    try:
        actor = MLPModel(
            TensorDict({"actor": torch.zeros(1, 140)}, batch_size=[1]),
            {"actor": ["actor"]},
            "actor",
            25,
            hidden_dims=[512, 256, 128],
            obs_normalization=True,
        ).eval()
        checkpoint = torch.load(
            demo.ASSETS_ROOT / "checkpoints/t800" / spec.asset_name / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        # Deterministic inference does not use the exploration distribution.
        state_dict = dict(checkpoint["actor_state_dict"])
        del state_dict["distribution.std_param"]
        actor.load_state_dict(state_dict, strict=True)
        env.init_state()
        obs, _ = env.reset(np.arange(num_envs, dtype=np.int32))
        resets = []
        with torch.inference_mode():
            for step in range(2 * spec.steps):
                action = actor(
                    TensorDict(
                        {"actor": torch.from_numpy(obs["obs"])}, batch_size=[num_envs]
                    )
                )
                state = env.step(action.numpy())
                obs = state.obs
                assert all(np.isfinite(v).all() for v in obs.values())
                assert not state.terminated.any(), f"{name} failed at step {step + 1}"
                if state.truncated.any():
                    assert state.truncated.all()
                    resets.append(step + 1)
        assert resets == [spec.steps, 2 * spec.steps]
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
