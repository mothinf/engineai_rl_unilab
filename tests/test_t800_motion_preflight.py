"""Training admission must reject invalid data before allocating an environment."""

from dataclasses import replace
import hashlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from engineai_rl_unilab.cli import _ensure_registry_env
from engineai_rl_unilab.tasks.t800.motion_preflight import (
    MotionValidationError,
    preflight_motion,
)
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter

from test_t800_motion_tracking import IDENTITY, owner

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ROOT / "assets/motions/t800"
V2 = MOTIONS / "dance1_subject2_t800_first18s_mujoco_v2.npz"
OLD = MOTIONS / "dance1_subject2_t800_first18s_mujoco.npz"


def config(path=V2):
    _ensure_registry_env()
    registry.ensure_registries()
    raw = owner()
    raw.env.commands.motion.params.motion_file = (
        [str(p) for p in path] if isinstance(path, list) else str(path)
    )
    override = BackendAdapter(
        raw, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    cfg = registry.materialize_env_config(IDENTITY)
    registry.apply_cfg_overrides(cfg, override)
    return cfg, override


def write_variant(tmp_path, problem):
    with np.load(V2, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if problem in ("joint_pos", "joint_vel"):
        values[problem][100:160, 5] += 1
    elif problem in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        values[problem][100:160, 7, 0] += 1
    elif problem == "body_quat_w":
        values[problem][100:160, 7] = [1, 0, 0, 0]
    elif problem == "nan":
        values["joint_pos"][0, 0] = np.nan
    elif problem == "missing_names":
        del values["body_names"]
    elif problem == "names_order":
        values["joint_names"][[0, 1]] = values["joint_names"][[1, 0]]
    elif problem == "unknown":
        values["unverified_field"] = np.zeros(900)
    elif problem == "short":
        for key in values:
            if key.startswith(("joint_", "body_")) and not key.endswith("names"):
                values[key] = values[key][:2]
    elif problem == "fps":
        values["fps"] = np.array([100])
    path = tmp_path / f"{problem}.npz"
    np.savez(path, **values)
    return path


def test_valid_file_passes_without_changing_config_or_npz(capsys):
    cfg, _ = config()
    before = hashlib.sha256(V2.read_bytes()).digest()
    checked = preflight_motion(cfg)
    assert checked is not cfg
    assert checked.commands["motion"] is not cfg.commands["motion"]
    assert checked.commands["motion"].motion_file == str(V2.resolve())
    assert cfg.commands["motion"].motion_file == str(V2)
    assert hashlib.sha256(V2.read_bytes()).digest() == before
    assert "[NPZ preflight] PASS" in capsys.readouterr().out


def test_old_file_blocks_registered_factory_before_allocation(monkeypatch):
    _, override = config(OLD)
    factory = Mock(side_effect=AssertionError("Must not allocate an environment"))
    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800_motion_tracking.make_manager_based_rl_env",
        factory,
    )
    with pytest.raises(MotionValidationError) as error:
        registry.make(
            IDENTITY, sim_backend="mujoco", num_envs=1024, env_cfg_override=override
        )
    message = str(error.value)
    for fragment in (str(OLD), "body_linear_fd", "LINK_ANKLE", "frame=", "limit="):
        assert fragment in message
    factory.assert_not_called()


@pytest.mark.parametrize(
    "problem",
    [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "nan",
        "missing_names",
        "names_order",
        "unknown",
        "short",
        "fps",
    ],
)
def test_invalid_fields_and_unverified_data_block_direct_factory(
    tmp_path, monkeypatch, problem
):
    from engineai_rl_unilab.tasks.t800_motion_tracking import (
        make_engineai_t800_motion_env,
    )

    path = write_variant(tmp_path, problem)
    cfg, _ = config(path)
    factory = Mock(side_effect=AssertionError("Must not allocate an environment"))
    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800_motion_tracking.make_manager_based_rl_env",
        factory,
    )
    with pytest.raises(MotionValidationError, match=path.name):
        make_engineai_t800_motion_env(cfg)
    factory.assert_not_called()


def test_isaac_requires_explicit_conversion(isaac_npz):
    cfg, _ = config(isaac_npz)
    with pytest.raises(MotionValidationError, match="engineai-convert-npz"):
        preflight_motion(cfg)


def test_every_clip_is_checked_and_not_concatenated(tmp_path):
    # Large inter-clip translation is fine: derivatives never cross boundaries.
    with np.load(V2) as archive:
        second = {key: archive[key].copy() for key in archive.files}
    second["body_pos_w"][:, 1:, 0] += 20
    path = tmp_path / "translated.npz"
    np.savez(path, **second)
    cfg, _ = config([V2, path])
    checked = preflight_motion(cfg)
    assert checked.commands["motion"].motion_file == [str(V2), str(path)]
    bad, _ = config([V2, OLD])
    with pytest.raises(MotionValidationError, match=OLD.name):
        preflight_motion(bad)


def test_resolved_paths_are_the_paths_passed_to_training(monkeypatch):
    cfg, _ = config("motions/t800/remote.npz")
    resolver = Mock(return_value=str(V2))
    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800.motion_preflight.resolve_motion_files", resolver
    )
    checked = preflight_motion(cfg)
    resolver.assert_called_once_with(cfg.commands["motion"].motion_file)
    assert checked.commands["motion"].motion_file == str(V2)
    assert cfg.commands["motion"].motion_file != str(V2)


def test_no_stale_pass_cache(tmp_path):
    path = tmp_path / "replaceable.npz"
    path.write_bytes(V2.read_bytes())
    cfg, _ = config(path)
    preflight_motion(cfg)
    path.write_bytes(OLD.read_bytes())
    with pytest.raises(MotionValidationError):
        preflight_motion(cfg)


def test_nominal_model_override_is_respected(tmp_path):
    from xml.etree import ElementTree

    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/robots/t800/scene_flat.xml")
    )
    model.body_pos[model.body("LINK_ANKLE_ROLL_L").id, 0] += 0.1
    path = tmp_path / "changed.xml"
    mujoco.mj_saveLastXML(str(path), model)
    # Saved XML flattens includes but keeps mesh paths relative to the old model.
    tree = ElementTree.parse(path)
    for node in tree.iter():
        if "file" in node.attrib:
            node.set("file", str(ROOT / "assets/robots/t800" / node.get("file")))
    tree.write(path)
    cfg, _ = config()
    cfg.scene = replace(cfg.scene, model_file=str(path))
    with pytest.raises(MotionValidationError, match="body_position_fk"):
        preflight_motion(cfg)


def test_empty_motion_list_is_not_a_pass():
    cfg, _ = config()
    cfg.commands["motion"].params.motion_file = []
    with pytest.raises(MotionValidationError, match="nonempty"):
        preflight_motion(cfg)
