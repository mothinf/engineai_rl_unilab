"""Regression cases for bad velocities that pose-only replay cannot reveal."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.tasks.motion_tracking.common.motion_loader import MotionData

from engineai_rl_unilab.tasks.t800.motion_npz import STATE_FIELDS
from engineai_rl_unilab.tasks.t800.motion_validation import (
    Metric,
    Tolerances,
    load_views,
    validate_motion,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
SOURCE = ROOT / "assets/motions/t800/dance1_subject2_t800_first18s.npz"
OLD = SOURCE.with_name("dance1_subject2_t800_first18s_mujoco.npz")
V2 = SOURCE.with_name("dance1_subject2_t800_first18s_mujoco_v2.npz")


@pytest.fixture(scope="module")
def names():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    return tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )


@pytest.fixture(scope="module")
def fixed(names):
    model, views = load_views(V2, MODEL, names, "mujoco")
    return model, views["file"]


def modified(view, frames=None):
    data = MotionData(
        **{k: getattr(view.data, k)[:frames].copy() for k in STATE_FIELDS}
    )
    return replace(view, data=data)


def test_old_fails_v2_passes_with_identical_pose(names, fixed):
    model, good = fixed
    _, views = load_views(OLD, MODEL, names, "mujoco")
    bad = views["file"]
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
        np.testing.assert_array_equal(getattr(bad.data, key), getattr(good.data, key))
    old_report, fixed_report = validate_motion(bad, model), validate_motion(good, model)
    assert old_report.status == "FAIL"
    assert fixed_report.status == "PASS"
    for name in ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"):
        i = good.body_names.index(name)
        old = old_report.metrics["body_linear_fd"]
        fixed_metric = fixed_report.metrics["body_linear_fd"]
        assert old.failed[i] and not fixed_metric.failed[i]
        assert old.rms[i] > 10 * fixed_metric.rms[i]
    assert 799 in old_report.anomaly_frames
    assert not len(fixed_report.anomaly_frames)


def test_source_is_retained_without_implicit_conversion(names, isaac_npz):
    digest = hashlib.sha256(isaac_npz.read_bytes()).digest()
    model, views = load_views(isaac_npz, MODEL, names, "t800_isaac_v1")
    assert set(views) == {"file"}
    with np.load(isaac_npz) as source:
        # Root is mapped to the target body ID; non-root states aren't repaired.
        np.testing.assert_array_equal(
            views["file"].data.body_lin_vel_w[:, 1], source["body_lin_vel_w"][:, 0]
        )
        np.testing.assert_array_equal(
            views["file"].data.body_lin_vel_w[:, 2], source["body_lin_vel_w"][:, 1]
        )
    raw = validate_motion(views["file"], model)
    assert raw.status == "PARTIAL" and not raw.complete
    assert np.isnan(raw.metrics["body_linear_fd"].error[:, 2:]).all()
    assert raw.metrics["body_angular_fd"].verified[1:].all()
    assert hashlib.sha256(isaac_npz.read_bytes()).digest() == digest
    json.dumps(raw.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "field,metric",
    [
        ("joint_pos", "body_position_fk"),
        ("joint_vel", "joint_velocity_fd"),
        ("body_pos_w", "body_position_fk"),
        ("body_quat_w", "body_orientation_fk"),
        ("body_lin_vel_w", "body_linear_fk"),
        ("body_ang_vel_w", "body_angular_fk"),
    ],
)
def test_each_state_field_corruption_is_located(fixed, field, metric):
    model, view = fixed
    view = modified(view, 30)
    value = getattr(view.data, field)
    if field == "body_quat_w":
        value[10, 7] = [1, 0, 0, 0]
    elif field.startswith("joint_"):
        value[10, 0] += 1
    else:
        value[10, 7, 0] += 1
    report = validate_motion(view, model)
    assert report.failed and report.metrics[metric].failed.any()
    assert 10 in report.anomaly_frames


def test_quaternion_sign_invariance_and_short_clip_boundaries(fixed):
    model, view = fixed
    flipped = modified(view)
    flipped.data.body_quat_w[::2] *= -1
    report = validate_motion(flipped, model)
    assert report.status == "PASS"
    assert np.isnan(report.angular_fd[[0, -1]]).all()
    for n in (1, 2):
        short = validate_motion(modified(view, n), model)
        assert short.status == "PARTIAL"
        assert not short.metrics["joint_velocity_fd"].verified.any()
        json.dumps(short.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "problem", ["body_order", "joint_order", "missing", "nan", "frames", "fps"]
)
def test_schema_and_legacy_layout_are_not_silently_repaired(tmp_path, names, problem):
    with np.load(V2) as data:
        payload = {k: data[k].copy() for k in data.files}
    if problem == "body_order":
        payload["body_names"][[1, 2]] = payload["body_names"][[2, 1]]
    elif problem == "joint_order":
        payload["joint_names"][[0, 1]] = payload["joint_names"][[1, 0]]
    elif problem == "missing":
        del payload["body_ang_vel_w"]
    elif problem == "nan":
        payload["body_lin_vel_w"][0, 1, 0] = np.nan
    elif problem == "frames":
        payload["body_quat_w"] = payload["body_quat_w"][:-1]
    else:
        payload["fps"] = np.array([-50])
    path = tmp_path / "bad.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        load_views(path, MODEL, names, "mujoco")


def test_joint_limits_and_tolerance_validation(fixed):
    model, view = fixed
    view = modified(view, 3)
    view.data.joint_pos[1, 0] = 99
    assert validate_motion(view, model).metrics["joint_limits"].failed[0]
    for value in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            Tolerances(linear_velocity=value)


def test_sustained_moderate_error_is_navigable(fixed):
    model, view = fixed
    report = validate_motion(view, model)
    metric = report.metrics["body_linear_fd"]
    error = np.zeros_like(metric.error)
    error[1:-1, 7] = 2 * metric.tolerance
    report.metrics["body_linear_fd"] = Metric(
        metric.name, metric.names, metric.units, error, metric.tolerance, True
    )
    assert report.failed
    assert 1 in report.anomaly_frames
    assert report.worst()[0] == "body_linear_fd"
