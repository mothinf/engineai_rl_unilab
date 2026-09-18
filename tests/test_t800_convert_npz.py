"""Offline conversion, non-destructive output and unchanged training-loader contract."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader

from engineai_rl_unilab.tasks.t800.convert_npz import convert_npz, main
from engineai_rl_unilab.tasks.t800.motion_npz import SOURCE_BODY_NAMES, STATE_FIELDS
from engineai_rl_unilab.tasks.t800.motion_validation import load_views, validate_motion
from engineai_rl_unilab.tasks.t800.replay import main as replay_main

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
V2 = ROOT / "assets/motions/t800/dance1_subject2_t800_first18s_mujoco_v2.npz"


def payload(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def test_roundtrip_shared_loader_and_replay(isaac_npz, tmp_path, monkeypatch):
    digest = hashlib.sha256(isaac_npz.read_bytes()).digest()
    output = tmp_path / "converted.npz"

    def no_physics(*args):
        pytest.fail("Conversion must not advance physics")

    monkeypatch.setattr(mujoco, "mj_step", no_physics)
    assert convert_npz(isaac_npz, output, model_file=MODEL) == output
    result, expected = payload(output), payload(V2)
    assert result.keys() == expected.keys()
    for key in ("fps", "joint_names", "body_names", "joint_pos", "joint_vel"):
        np.testing.assert_array_equal(result[key], expected[key])
    for key in STATE_FIELDS:
        assert result[key].dtype == np.float32
        # Re-running FK from saved float32 root states adds rounding; velocity
        # composition is particularly sensitive to the rounded quaternion.
        np.testing.assert_allclose(result[key], expected[key], atol=1e-5, rtol=0)
    loader = MotionLoader(str(output))
    for key in STATE_FIELDS:
        np.testing.assert_array_equal(getattr(loader, key), result[key])
    selected = np.array([1, 7, 14])
    subset = MotionLoader(str(output), body_indices=selected)
    np.testing.assert_array_equal(
        subset.body_lin_vel_w, result["body_lin_vel_w"][:, selected]
    )
    model, views = load_views(output, MODEL, tuple(result["joint_names"]), "mujoco")
    assert validate_motion(views["file"], model).status == "PASS"
    assert replay_main(["--npz-file", str(output), "--check-only"]) == 0
    assert hashlib.sha256(isaac_npz.read_bytes()).digest() == digest


def test_nonroot_source_data_is_explicitly_rebuilt(isaac_npz, tmp_path):
    original = payload(isaac_npz)
    changed = {key: value.copy() for key, value in original.items()}
    changed["body_pos_w"][:, 1:] += 10
    changed["body_quat_w"][:, 1:] = [1, 0, 0, 0]
    changed["body_lin_vel_w"][:, 1:] += 20
    changed["body_ang_vel_w"][:, 1:] -= 20
    # Valid optional metadata follows the same declared source layout.
    changed["body_names"] = np.array(SOURCE_BODY_NAMES)
    source = tmp_path / "changed.npz"
    np.savez(source, **changed)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    convert_npz(isaac_npz, a, model_file=MODEL)
    convert_npz(source, b, model_file=MODEL)
    left, right = payload(a), payload(b)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_world_root_velocity_preserved_without_joint_clipping(isaac_npz, tmp_path):
    values = {
        key: value[:12] if key in STATE_FIELDS else value
        for key, value in payload(isaac_npz).items()
    }
    values["body_quat_w"][:, 0] = [0.5, 0.5, 0.5, 0.5]
    values["body_lin_vel_w"][:, 0] = [1.2, -0.4, 0.7]
    values["body_ang_vel_w"][:, 0] = [-0.8, 0.3, 1.6]
    values["joint_pos"][:, 0] = 12
    values["joint_vel"][:, 0] = 30
    values["fps"] = np.array([100])
    source, output = tmp_path / "root.npz", tmp_path / "output.npz"
    np.savez(source, **values)
    convert_npz(source, output, model_file=MODEL)
    result = payload(output)
    assert result["fps"][0] == 100 and result["joint_pos"].shape == (12, 25)
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_allclose(result[field][:, 1], values[field][:, 0], atol=1e-6)
    joint = result["joint_names"].tolist().index(values["joint_names"][0])
    np.testing.assert_array_equal(result["joint_pos"][:, joint], 12)
    np.testing.assert_array_equal(result["joint_vel"][:, joint], 30)
    for field in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_array_equal(result[field][:, 0], 0)
    np.testing.assert_array_equal(
        result["body_quat_w"][:, 0], np.tile([1, 0, 0, 0], (12, 1))
    )


@pytest.mark.parametrize(
    "problem",
    [
        "fps",
        "missing",
        "nan",
        "quaternion",
        "body_order",
        "joint_name",
        "joint_duplicates",
        "extra",
        "frames",
        "mujoco_input",
    ],
)
def test_bad_source_fails_before_writing(isaac_npz, tmp_path, problem):
    values = payload(isaac_npz)
    if problem == "fps":
        values["fps"] = np.array([49.5])
    elif problem == "missing":
        del values["joint_names"]
    elif problem == "nan":
        values["body_lin_vel_w"][0, 2, 0] = np.nan
    elif problem == "quaternion":
        values["body_quat_w"][0, 2] = 0
    elif problem == "body_order":
        values["body_names"] = np.array(SOURCE_BODY_NAMES[::-1])
    elif problem == "joint_name":
        values["joint_names"][0] = "not_a_joint"
    elif problem == "joint_duplicates":
        values["joint_names"][0] = values["joint_names"][1]
    elif problem == "extra":
        values["unhandled"] = np.zeros(10)
    elif problem == "frames":
        values["joint_vel"] = values["joint_vel"][:-1]
    else:
        values = payload(V2)
    source, output = tmp_path / "bad.npz", tmp_path / "output.npz"
    np.savez(source, **values)
    with pytest.raises(ValueError):
        convert_npz(source, output, model_file=MODEL)
    assert not output.exists()


def test_no_overwrite_and_failed_write_cleanup(isaac_npz, tmp_path, monkeypatch):
    before = isaac_npz.read_bytes()
    with pytest.raises(ValueError, match="different"):
        convert_npz(isaac_npz, isaac_npz, model_file=MODEL)
    output = tmp_path / "existing.npz"
    output.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        convert_npz(isaac_npz, output, model_file=MODEL)
    assert output.read_bytes() == b"user file" and isaac_npz.read_bytes() == before
    with pytest.raises(ValueError, match="extension"):
        convert_npz(isaac_npz, tmp_path / "no_suffix", model_file=MODEL)

    def interrupted(stream, **kwargs):
        stream.write(b"incomplete")
        raise OSError("disk write failed")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    partial = tmp_path / "partial.npz"
    with pytest.raises(OSError, match="disk write failed"):
        convert_npz(isaac_npz, partial, model_file=MODEL)
    assert not partial.exists()


def test_cli(isaac_npz, tmp_path, capsys):
    output = tmp_path / "cli.npz"
    args = [
        "--input",
        str(isaac_npz),
        "--output",
        str(output),
        "--model-file",
        str(MODEL),
    ]
    assert main(args) == 0
    assert "Input unchanged" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_converted_motion_runs_in_unchanged_training(isaac_npz, tmp_path, monkeypatch):
    from test_t800_motion_tracking import IDENTITY, owner
    from engineai_rl_unilab.cli import _ensure_registry_env
    from unilab.base import registry
    from unilab.base.config_adapter import BackendAdapter

    output = tmp_path / "train.npz"
    convert_npz(isaac_npz, output, model_file=MODEL)
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner()
    assert "motion_adapter" not in cfg.env.commands.motion
    cfg.env.commands.motion.params.motion_file = str(output)
    override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()

    def forbidden(*args, **kwargs):
        pytest.fail("Training must not invoke offline conversion or tool FK")

    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800.convert_npz.convert_npz", forbidden
    )
    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800.motion_npz.tracking_fk", forbidden
    )
    env = registry.make(
        IDENTITY, num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.init_state()
        assert type(env.command_manager.get_term("motion").motion) is MotionLoader
        env.reset(np.arange(2, dtype=np.int32))
        assert env.obs_groups_spec == {"obs": 134, "critic": 275}
        for _ in range(3):
            state = env.step(np.zeros((2, 25), dtype=np.float32))
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(value).all() for value in state.obs.values())
    finally:
        env.close()
