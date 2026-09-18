"""Independent reconstructions, analytical oracles and six-field corruption."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.tasks.motion_tracking.common.motion_loader import MotionData

from engineai_rl_unilab.tasks.t800.motion_npz import STATE_FIELDS
from engineai_rl_unilab.tasks.t800.motion_ghosts import (
    GhostOptions,
    build_ghosts,
    integrate_linear,
    integrate_world_orientation,
    orientation_error,
    quat_mul,
)
from engineai_rl_unilab.tasks.t800.motion_validation import load_views, validate_motion

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
MOTIONS = ROOT / "assets/motions/t800"
V2 = MOTIONS / "dance1_subject2_t800_first18s_mujoco_v2.npz"
OLD = MOTIONS / "dance1_subject2_t800_first18s_mujoco.npz"


@pytest.fixture(scope="module")
def loaded():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    _, views = load_views(V2, MODEL, names, "mujoco")
    view = views["file"]
    return model, names, view, build_ghosts(validate_motion(view, model), model)


def clone(view, frames=None):
    return replace(
        view,
        data=MotionData(
            **{key: getattr(view.data, key)[:frames].copy() for key in STATE_FIELDS}
        ),
    )


def test_old_v2_spatial_acceptance(loaded):
    model, names, view, good = loaded
    _, old_views = load_views(OLD, MODEL, names, "mujoco")
    bad = build_ghosts(validate_motion(old_views["file"], model), model)
    assert good.status == "PASS" and bad.status == "FAIL"
    np.testing.assert_array_equal(good.base.fk.body_pos_w, bad.base.fk.body_pos_w)
    np.testing.assert_array_equal(good.base.fk.body_quat_w, bad.base.fk.body_quat_w)
    ankle = view.body_names.index("LINK_ANKLE_ROLL_L")
    assert bad.ghosts["body-velocity"].position_error[799, ankle] > 0.2
    assert good.ghosts["body-velocity"].position_error[799, ankle] < 0.01
    assert np.nanmax(bad.ghosts["body-velocity"].position_error) == pytest.approx(
        0.432271, abs=1e-5
    )
    assert np.nanmax(good.ghosts["body-velocity"].position_error) == pytest.approx(
        0.0183635, abs=1e-6
    )
    assert np.rad2deg(
        np.nanmax(good.ghosts["body-velocity"].angle_error)
    ) == pytest.approx(4.6596, abs=0.001)
    assert np.nanmax(good.ghosts["joint-velocity"].position_error) < 0.03
    assert not len(good.anomaly_frames) and 799 in bad.anomaly_frames
    assert 0 <= bad.worst()[0] < 900 and bad.worst()[1] > 0


@pytest.mark.parametrize(
    "field,layer",
    [
        ("joint_pos", "pose"),
        ("joint_vel", "joint-velocity"),
        ("body_pos_w", "pose"),
        ("body_quat_w", "pose"),
        ("body_lin_vel_w", "body-velocity"),
        ("body_ang_vel_w", "body-velocity"),
    ],
)
def test_six_fields_produce_localizable_ghost_errors(loaded, field, layer):
    model, _, original, _ = loaded
    view = clone(original, 60)
    body = view.body_names.index("LINK_ANKLE_ROLL_L")
    joint = view.joint_names.index("J05_ANKLE_ROLL_L")
    value = getattr(view.data, field)
    if field == "body_quat_w":
        delta = np.array([np.cos(0.25), 0, np.sin(0.25), 0])
        value[10:40, body] = quat_mul(delta, value[10:40, body])
    elif field.startswith("joint_"):
        value[10:40, joint] += 0.5 if field == "joint_pos" else 2
    else:
        value[10:40, body, 0] += 0.2 if field == "body_pos_w" else 2
    report = build_ghosts(validate_motion(view, model), model)
    ghost = report.ghosts[layer]
    assert report.failed and ghost.hotspots[25, body]
    assert 25 in report.anomaly_frames
    assert ghost.drawable[25, body]
    # Mutating a view for a test must not have modified the original input.
    assert not np.array_equal(value, getattr(original.data, field)[:60])


def test_pose_not_projected_and_joint_ghost_not_clipped(loaded):
    model, _, original, _ = loaded
    view = clone(original, 40)
    view.data.body_pos_w[:, 7, 0] += 0.4
    view.data.joint_vel[:, 5] = 30
    report = build_ghosts(validate_motion(view, model), model)
    np.testing.assert_array_equal(report.ghosts["pose"].position, view.data.body_pos_w)
    # A velocity-integrated hinge is allowed to pass its physical limit. Verify
    # geometry against an explicit FK of the un-clipped angle.
    frame = 20
    data = mujoco.MjData(model)
    data.qpos[:3] = view.data.body_pos_w[frame, 1]
    data.qpos[3:7] = view.data.body_quat_w[frame, 1]
    joint_ids = [model.joint(name).id for name in view.joint_names]
    q = integrate_linear(view.data.joint_pos, view.data.joint_vel, 10, 0.02)[frame]
    assert q[5] > model.jnt_range[joint_ids[5], 1]
    data.qpos[model.jnt_qposadr[joint_ids]] = q
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        report.ghosts["joint-velocity"].position[frame], data.xpos, atol=1e-6
    )


@pytest.mark.parametrize("fps", [25, 50, 100])
def test_analytical_translation_and_world_rotation(fps):
    n, steps, dt = 41, round(0.2 * fps), 1 / fps
    t = np.arange(n) * dt
    velocity = np.broadcast_to([0.3, -0.7, 1.2], (n, 1, 3))
    p = np.array([1, 2, 3]) + t[:, None, None] * velocity
    np.testing.assert_allclose(
        integrate_linear(p, velocity, steps, dt)[steps:], p[steps:], atol=1e-12
    )
    initial = np.array([np.cos(0.4), np.sin(0.4), 0, 0])
    q = np.array(
        [quat_mul(np.array([np.cos(x), 0, 0, np.sin(x)]), initial) for x in t]
    )[:, None]
    omega = np.broadcast_to([0.0, 0.0, 2.0], (n, 1, 3))
    predicted = integrate_world_orientation(q, omega, steps, dt)
    assert np.isnan(predicted[:steps]).all()
    np.testing.assert_allclose(
        orientation_error(predicted[steps:], q[steps:]), 0, atol=1e-12
    )


def test_noncommuting_world_rotations_against_mujoco_oracle():
    q0 = np.array([np.cos(0.3), np.sin(0.3), 0, 0])
    q = np.broadcast_to(q0, (3, 1, 4)).copy()
    omega = np.array([[[1.0, 0, 0]], [[0, 2.0, 0]], [[0, 0, 3.0]]])
    expected = q0.copy()
    deltas = []
    for i in range(2):
        rotation = 0.1 * (omega[i, 0] + omega[i + 1, 0]) / 2
        delta, result = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(
            delta, rotation / np.linalg.norm(rotation), np.linalg.norm(rotation)
        )
        mujoco.mju_mulQuat(result, delta, expected)
        expected = result
        deltas.append(delta)
    predicted = integrate_world_orientation(q, omega, 2, 0.1)[2, 0]
    np.testing.assert_allclose(predicted, expected, atol=1e-12)
    wrong_order = quat_mul(deltas[0], quat_mul(deltas[1], q0))
    assert orientation_error(predicted, wrong_order) > 0.001


def test_warmup_random_access_and_quaternion_sign(loaded):
    model, _, original, good = loaded
    flipped = clone(original)
    flipped.data.body_quat_w[::2] *= -1
    result = build_ghosts(validate_motion(flipped, model), model)
    for name, ghost in result.ghosts.items():
        np.testing.assert_allclose(
            ghost.position_error, good.ghosts[name].position_error, atol=1e-6
        )
        np.testing.assert_allclose(
            ghost.angle_error, good.ghosts[name].angle_error, atol=1e-6
        )
        if name != "pose":
            assert not ghost.drawable[:10].any() and ghost.drawable[10, 1]
    for n in (1, 2, 10):
        short = clone(original, n)
        for field in STATE_FIELDS:
            values = getattr(short.data, field)
            values[:] = 0 if "vel" in field else values[0]
        report = build_ghosts(validate_motion(short, model), model)
        assert report.status == "PARTIAL"
        assert not report.ghosts["body-velocity"].drawable.any()
        assert not report.ghosts["joint-velocity"].drawable.any()
        json.dumps(report.as_dict(), allow_nan=False)
    assert np.isnan(good.ghosts["pose"].position_error[:, 1]).all()
    assert np.isnan(good.ghosts["joint-velocity"].angle_error[:, 1]).all()
    assert np.isfinite(good.ghosts["body-velocity"].position_error[10:, 1]).all()


def test_raw_source_unknown_mapping_and_file_unchanged(loaded, isaac_npz):
    model, names, _, _ = loaded
    digest = hashlib.sha256(isaac_npz.read_bytes()).hexdigest()
    _, views = load_views(isaac_npz, MODEL, names, "t800_isaac_v1")
    raw = build_ghosts(validate_motion(views["file"], model), model)
    assert raw.status == "PARTIAL"
    for name in ("pose", "body-velocity"):
        assert not raw.ghosts[name].drawable[:, 2:].any()
        assert np.isnan(raw.ghosts[name].position_error[:, 2:]).all()
    assert raw.ghosts["joint-velocity"].drawable[10:, 2:].all()
    assert set(views) == {"file"}
    assert hashlib.sha256(isaac_npz.read_bytes()).hexdigest() == digest


def test_coverage_options_and_window_rounding(loaded):
    model, _, view, good = loaded
    report = build_ghosts(
        good.base,
        model,
        GhostOptions(window=0.211),
        ("mystery", *STATE_FIELDS, "fps", "joint_names", "body_names"),
    )
    assert report.window_frames == 11
    assert report.status == "PARTIAL" and report.unsupported_fields == ("mystery",)
    payload = report.as_dict()
    assert payload["ghosts"]["actual_window_s"] == 0.22
    assert set(payload["coverage"]["uses"]) == {
        *STATE_FIELDS,
        "fps",
        "joint_names",
        "body_names",
    }
    json.dumps(payload, allow_nan=False)
    # Changing only fps changes integration time, not a hidden simulator dt.
    slow = replace(view, fps=25)
    changed = build_ghosts(validate_motion(slow, model), model)
    assert changed.window_frames == 5 and changed.failed
    for key in ("window", "position_tol", "angle_tol_deg"):
        for value in (0, -1, np.nan, np.inf):
            with pytest.raises(ValueError):
                GhostOptions(**{key: value})
    with pytest.raises(ValueError, match="window is too large"):
        build_ghosts(good.base, model, GhostOptions(window=1e308))
