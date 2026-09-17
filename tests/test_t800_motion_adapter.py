"""Numerical and compatibility boundaries of the opt-in T800 adapter."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from engineai_rl_unilab.tasks.t800.manager_terms import OfficialMotionCommand
from engineai_rl_unilab.tasks.t800.motion_adapter import (
    STATE_FIELDS,
    _tracking_fk,
    adapt_t800_isaac_motion,
)
from engineai_rl_unilab.tasks.t800.motion_loader import T800MotionLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
SOURCE = ROOT / "assets/motions/t800/dance1_subject2_t800_first18s.npz"
V2 = SOURCE.with_name("dance1_subject2_t800_first18s_mujoco_v2.npz")
SOURCE_SHA256 = "02bf4bf478b71dca6b06ecc34281105362d9f51e117dd9f0db53cae69e674a57"


@pytest.fixture(scope="module")
def joint_names():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    return tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )


def test_source_unchanged_and_v2_parity(joint_names):
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    adapted = adapt_t800_isaac_motion(SOURCE, model_file=MODEL, joint_names=joint_names)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before == SOURCE_SHA256
    assert adapted.fps == 50
    with np.load(V2) as expected:
        for field in STATE_FIELDS:
            result = getattr(adapted.data, field)
            assert result.dtype == np.float32
            assert np.isfinite(result).all()
            if field.startswith("joint_"):
                np.testing.assert_array_equal(result, expected[field])
            else:
                np.testing.assert_allclose(
                    result, expected[field], atol=1e-5, rtol=1e-5
                )


def test_verified_source_root_velocity_is_origin_velocity():
    # Pin the source-specific assumption; the Isaac API name alone is ambiguous.
    with np.load(SOURCE) as source:
        fps = float(source["fps"].reshape(-1)[0])
        positions = source["body_pos_w"][:, 0]
        derivative = np.gradient(positions.astype(np.float64), 1 / fps, axis=0)
        # Stored float32 world positions lose precision before differentiation.
        quantization_bound = float(np.abs(np.spacing(positions)).max()) * fps
        np.testing.assert_allclose(
            source["body_lin_vel_w"][1:-1, 0],
            derivative[1:-1],
            atol=quantization_bound,
            rtol=0,
        )
        error = source["body_lin_vel_w"][1:-1, 0] - derivative[1:-1]
        assert np.sqrt(np.mean(error**2)) < 2e-6


def test_rotated_root_velocity_at_body_origin():
    # Root and child COM offsets deliberately differ from the body origins.
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="LINK_BASE"><freejoint/>
        <inertial pos="0.3 0.2 0.1" mass="1" diaginertia="1 1 1"/>
        <body name="child" pos="2 0 0"><joint name="hinge" axis="0 0 1"/>
          <inertial pos="0.5 0.2 0.1" mass="1" diaginertia="1 1 1"/>
        </body></body></worldbody>
        <sensor><framelinvel name="child_lin" objtype="xbody" objname="child"/>
        <frameangvel name="child_ang" objtype="xbody" objname="child"/></sensor>
      </mujoco>""")
    root_velocity = np.array([0.2, -0.1, 0.3])
    omega = np.array([1.0, 0.0, 0.0])
    source = dict(
        joint_pos=np.zeros((1, 1)),
        joint_vel=np.zeros((1, 1)),
        body_pos_w=np.array([[[0.0, 0.0, 1.0]]]),
        body_quat_w=np.array([[[2**-0.5, 0.0, 0.0, 2**-0.5]]]),
        body_lin_vel_w=root_velocity[None, None],
        body_ang_vel_w=omega[None, None],
    )
    result = _tracking_fk(model, ("hinge",), source)
    np.testing.assert_allclose(result.body_ang_vel_w[0, 1:], [omega, omega], atol=1e-6)
    np.testing.assert_allclose(result.body_lin_vel_w[0, 1], root_velocity, atol=1e-6)
    expected = root_velocity + np.cross(omega, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(result.body_lin_vel_w[0, 2], expected, atol=1e-6)
    # Independent, body-frame sensor readback used by the training backend.
    data = mujoco.MjData(model)
    data.qpos[:3] = source["body_pos_w"][0, 0]
    data.qpos[3:7] = source["body_quat_w"][0, 0]
    data.qvel[:3] = root_velocity
    data.qvel[3:6] = [0.0, -1.0, 0.0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        result.body_lin_vel_w[0, 2], data.sensor("child_lin").data, atol=1e-6
    )
    np.testing.assert_allclose(
        result.body_ang_vel_w[0, 2], data.sensor("child_ang").data, atol=1e-6
    )


def test_names_drive_joint_order_and_nonroot_states_are_regenerated(
    tmp_path, joint_names
):
    with np.load(SOURCE) as source:
        payload = {k: source[k].copy() for k in source.files}
    for field in STATE_FIELDS:
        payload[field] = payload[field][:2].copy()
    order = np.arange(25)[::-1]
    payload["joint_names"] = payload["joint_names"][order]
    for field in ("joint_pos", "joint_vel"):
        payload[field] = payload[field][:, order]
    for field in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        payload[field][:, 1:] = 123.0  # Not accepted as target body references.
    altered = tmp_path / "shuffled.npz"
    np.savez(altered, **payload)
    adapted = adapt_t800_isaac_motion(
        altered, model_file=MODEL, joint_names=joint_names
    )
    with np.load(V2) as expected:
        for field in STATE_FIELDS:
            np.testing.assert_allclose(
                getattr(adapted.data, field), expected[field][:2], atol=1e-5
            )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "frames",
        "nan",
        "quaternion",
        "fps",
        "body_names",
        "empty",
    ],
)
def test_rejects_invalid_source(tmp_path, joint_names, problem):
    with np.load(SOURCE) as source:
        payload = {k: source[k].copy() for k in source.files}
    if problem == "missing":
        del payload["joint_vel"]
    elif problem == "duplicate":
        payload["joint_names"][0] = payload["joint_names"][1]
    elif problem == "frames":
        payload["body_pos_w"] = payload["body_pos_w"][:-1]
    elif problem == "nan":
        payload["joint_vel"][0, 0] = np.nan
    elif problem == "quaternion":
        payload["body_quat_w"][0, 0] = 0
    elif problem == "fps":
        payload["fps"] = np.array([0])
    elif problem == "body_names":
        payload["body_names"] = np.array(["wrong"] * 30)
    else:
        payload["joint_pos"] = payload["joint_pos"][:0]
    path = tmp_path / "invalid.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        adapt_t800_isaac_motion(path, model_file=MODEL, joint_names=joint_names)


def test_rejects_converted_npz_and_unknown_adapter(joint_names):
    with pytest.raises(ValueError, match="body_names"):
        adapt_t800_isaac_motion(V2, model_file=MODEL, joint_names=joint_names)
    command = object.__new__(OfficialMotionCommand)
    command.cfg = SimpleNamespace(motion_adapter="typo")
    with pytest.raises(ValueError, match="Unsupported motion_adapter"):
        command._make_motion_loader(str(SOURCE), np.array([1]))


def test_multiclip_selection_and_buffer_gather(tmp_path, joint_names):
    with np.load(SOURCE) as source:
        payload = {k: source[k].copy() for k in source.files}
    for field in STATE_FIELDS:
        payload[field] = payload[field][:3]
    path = tmp_path / "short.npz"
    np.savez(path, **payload)
    loader = T800MotionLoader(
        [str(path), str(path)],
        model_file=str(MODEL),
        joint_names=joint_names,
        body_indices=np.array([1, 7]),
    )
    assert loader.num_clips == 2 and loader.num_frames == 6
    np.testing.assert_array_equal(loader.clip_offsets, [0, 3])
    np.testing.assert_array_equal(loader.clip_end_frames, [2, 5])
    np.testing.assert_array_equal(loader.get_clip_indices(np.array([2, 3])), [0, 1])
    out = loader.make_motion_data_buffer(2)
    assert loader.get_motion_at_frame(np.array([0, 3]), out=out) is out
    assert out.body_pos_w.shape == (2, 2, 3)
    for field in STATE_FIELDS:
        np.testing.assert_array_equal(getattr(out, field)[0], getattr(out, field)[1])
    payload["fps"] = np.array([25])
    other = tmp_path / "other_fps.npz"
    np.savez(other, **payload)
    with pytest.raises(ValueError, match="same fps"):
        T800MotionLoader(
            [str(path), str(other)], model_file=str(MODEL), joint_names=joint_names
        )
