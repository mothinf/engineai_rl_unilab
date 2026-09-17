"""Physical stepping and the actual diagnostics sent to the viewer."""

from pathlib import Path
from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from engineai_rl_unilab.tasks.t800.motion_validation import load_views, validate_motion
from engineai_rl_unilab.tasks.t800.replay import (
    CYAN,
    KEY_BINDINGS,
    KEY_HELP,
    RED,
    YELLOW,
    ReplayState,
    check_viewer_shortcuts,
    diagnostic_figures,
    diagnostic_texts,
    draw_diagnostics,
    main,
    overlay_text,
    run_viewer,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
DIR = ROOT / "assets/motions/t800"
OLD = DIR / "dance1_subject2_t800_first18s_mujoco.npz"
V2 = DIR / "dance1_subject2_t800_first18s_mujoco_v2.npz"


def load(path=V2):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    model, views = load_views(path, MODEL, names, "mujoco")
    return model, views["file"]


def test_dynamic_steps_without_overwriting_state(monkeypatch):
    model, view = load()
    state = ReplayState(model, view, "dynamic")
    state.initialize(0)
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(
        model, state.data, mujoco.mjtObj.mjOBJ_XBODY, state.root, velocity, 0
    )
    np.testing.assert_allclose(
        velocity[:3], view.data.body_ang_vel_w[0, state.root], atol=1e-6
    )
    np.testing.assert_allclose(
        velocity[3:], view.data.body_lin_vel_w[0, state.root], atol=1e-6
    )
    # A perturbation must survive into physical integration, not be overwritten.
    state.data.qpos[state.root_qadr + 2] += 0.2

    def forbidden(*args):
        pytest.fail("Dynamic advance wrote reference state")

    monkeypatch.setattr(state, "set_frame", forbidden)
    for _ in range(5):
        assert state.advance()
    assert state.initializations == 1
    assert state.frame == 5 and state.data.time == pytest.approx(0.1)
    assert (
        np.linalg.norm(
            state.data.xpos[state.root] - view.data.body_pos_w[5, state.root]
        )
        > 0.05
    )
    assert np.isfinite(state.data.qpos).all()


def test_kinematic_frame_and_dynamic_timestep_contract():
    model, view = load()
    state = ReplayState(model, view)
    state.initialize(798)
    assert state.advance()
    np.testing.assert_allclose(state.data.qpos[state.qadr], view.data.joint_pos[799])
    np.testing.assert_allclose(state.data.xpos, view.data.body_pos_w[799], atol=1e-6)
    state.set_frame(len(view.data.joint_pos) - 1)
    assert not state.advance()
    for dt in (0, -1, 0.003, np.nan):
        with pytest.raises(ValueError):
            ReplayState(model, view, "dynamic", dt)


def test_viewer_marks_bad_ankle_and_preserves_arrow_magnitude():
    model, view = load(OLD)
    report = validate_motion(view, model)
    state = ReplayState(model, view)
    state.initialize(799)
    body = model.body("LINK_ANKLE_ROLL_L").id
    scene = mujoco.MjvScene(model, maxgeom=1000)
    draw_diagnostics(scene, state, report, body, "linear", vector_scale=0.2)
    marks = [g for g in scene.geoms[: scene.ngeom] if g.label == "LINK_ANKLE_ROLL_L"]
    assert len(marks) == 1
    np.testing.assert_allclose(marks[0].rgba, RED)
    arrows = [
        g for g in scene.geoms[: scene.ngeom] if g.type == mujoco.mjtGeom.mjGEOM_ARROW
    ]
    assert len(arrows) == 2
    np.testing.assert_allclose(arrows[0].rgba, CYAN)
    np.testing.assert_allclose(arrows[1].rgba, YELLOW)
    # Connector lengths must retain the NPZ vs differentiated speed ratio.
    lengths = [g.size[2] for g in arrows]
    expected_ratio = np.linalg.norm(
        view.data.body_lin_vel_w[799, body]
    ) / np.linalg.norm(report.linear_fd[799, body])
    assert lengths[0] / lengths[1] == pytest.approx(expected_ratio, rel=1e-5)
    assert "FAIL" in overlay_text(state, report, body, 0, "linear", True)
    for layer in ("position", "orientation", "linear", "angular"):
        scene.ngeom = 0
        draw_diagnostics(scene, state, report, body, layer)
        figures = diagnostic_figures(report, 799, body, 0, layer)
        assert len(figures) == 2
        assert figures[0][1].linepnt[0] > 0
        assert figures[0][1].range[0, 0] <= 799 / view.fps <= figures[0][1].range[0, 1]
        for _, _, text1, text2 in diagnostic_texts(state, report, body, 0, layer, True):
            assert len(text1) < mujoco.mjMAXOVERLAY
            assert len(text2) < mujoco.mjMAXOVERLAY


def test_cli_report_exit_codes_and_default_adapter(tmp_path, capsys):
    report = tmp_path / "report.json"
    args = ["--format", "mujoco", "--check-only", "--report", str(report)]
    assert main([*args, "--npz-file", str(OLD)]) == 1
    data = json.loads(report.read_text())
    assert data["file"]["status"] == "FAIL"
    assert (
        data["file"]["metrics"]["body_linear_fd"]["entities"]["LINK_ANKLE_ROLL_L"][
            "status"
        ]
        == "FAIL"
    )
    assert main([*args, "--npz-file", str(V2)]) == 0
    assert json.loads(report.read_text())["file"]["status"] == "PASS"
    assert main(["--check-only", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["file"]["status"] == "PARTIAL"
    assert data["reference"]["status"] == "PASS"
    assert main(["--headless", "--mode", "dynamic", "--steps", "3"]) == 0
    assert "initializations=1" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--check-only", "--npz-file", str(OLD)])  # Explicit format required.
    assert exc.value.code == 2


def test_bindings_do_not_overlap_installed_mujoco_shortcuts(monkeypatch):
    # Check the actual runtime tables, including the tempting but occupied K/L.
    check_viewer_shortcuts()
    native = {
        ord(key[0]) for _, _, key in (*mujoco.mjVISSTRING, *mujoco.mjRNDSTRING) if key
    }
    assert {ord("J"), ord("B"), ord("K"), ord("L"), ord("T")} <= native
    assert native.isdisjoint(KEY_BINDINGS)
    assert set(range(ord("0"), ord("5") + 1)).isdisjoint(KEY_BINDINGS)
    # A future accidental reintroduction must fail before opening the viewer.
    monkeypatch.setitem(KEY_BINDINGS, ord("J"), "joint")
    with pytest.raises(RuntimeError, match="conflict"):
        check_viewer_shortcuts()


def run_viewer_keys(monkeypatch, model, reports, batches, **options):
    """Exercise the real event loop without requiring an X11/Wayland server."""
    import mujoco.viewer

    native_vis = {
        ord(key[0]): i for i, (_, _, key) in enumerate(mujoco.mjVISSTRING) if key
    }
    native_rnd = {
        ord(key[0]): i for i, (_, _, key) in enumerate(mujoco.mjRNDSTRING) if key
    }

    class Viewer:
        def __init__(self):
            self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
            self.user_scn = mujoco.MjvScene(model, maxgeom=1000)
            self.viewport = mujoco.MjrRect(0, 0, 1200, 900)
            self.index, self.snapshots = 0, []

        def is_running(self):
            if self.index == len(batches):
                return False
            for key in batches[self.index]:
                # MuJoCo handles the native action first; then invokes Python.
                if key in native_vis:
                    i = native_vis[key]
                    self.opt.flags[i] = not self.opt.flags[i]
                if key in native_rnd:
                    i = native_rnd[key]
                    self.user_scn.flags[i] = not self.user_scn.flags[i]
                if ord("0") <= key <= ord("5"):
                    self.opt.geomgroup[key - ord("0")] ^= 1
                self.callback(key)
            self.index += 1
            return True

        def lock(self):
            return nullcontext()

        def set_texts(self, texts):
            for _, _, left, right in texts:
                assert len(left) < mujoco.mjMAXOVERLAY
                assert len(right) < mujoco.mjMAXOVERLAY
            self.texts = "\n".join(text for _, _, a, b in texts for text in (a, b))

        def set_figures(self, figures):
            self.figures = figures

        def sync(self):
            self.snapshots.append(
                (
                    self.texts,
                    len(self.figures),
                    self.opt.flags.copy(),
                    self.opt.geomgroup.copy(),
                )
            )

    viewer = Viewer()

    def launch(model, data, key_callback):
        viewer.callback = key_callback
        return nullcontext(viewer)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(
        "engineai_rl_unilab.tasks.t800.replay.time.sleep", lambda _: None
    )
    args = SimpleNamespace(
        view="file",
        mode="kinematic",
        sim_dt=1 / 150,
        start_frame=100,
        body="LINK_ANKLE_ROLL_L",
        joint="J05_ANKLE_ROLL_L",
        play=False,
        vector_scale=0.2,
        speed=1,
        loop=False,
    )
    vars(args).update(options)
    run_viewer(model, reports, args)
    return viewer.snapshots


def test_viewer_key_dispatch_and_native_keys_are_independent(monkeypatch):
    model, view = load(OLD)
    reports = {
        key: validate_motion(replace(view, label=key), model)
        for key in ("file", "reference")
    }
    # Initial frame, function keys, navigation, then native letter/group keys.
    batches = [
        [],
        [297],
        [298],
        [299],
        [299],
        [299],
        [299],
        [300],
        [301],
        [301],
        [262],
        [263],
        [ord("7")],
        [ord("6")],
        [ord("8")],
        [ord(key) for key in "BJKLNPRVG1234T"],
        [32],
        [],
        [32],
    ]
    snapshots = run_viewer_keys(monkeypatch, model, reports, batches)
    assert "J06_HIP_PITCH_R" in snapshots[1][0]
    assert "Body: LINK_FOOT_L" in snapshots[2][0]
    for i, layer in enumerate(("angular", "position", "orientation", "linear"), 3):
        assert f"Body: LINK_FOOT_L | {layer}" in snapshots[i][0]
    assert "FAIL | reference" in snapshots[7][0]
    assert snapshots[8][1] == 0 and snapshots[9][1] == 2
    assert "frame 101/" in snapshots[10][0]
    assert "frame 100/" in snapshots[11][0]
    assert "frame 106/" in snapshots[12][0] and "PAUSED" in snapshots[12][0]
    assert "frame 94/" in snapshots[13][0] and "PAUSED" in snapshots[13][0]
    assert "frame 100/" in snapshots[14][0]
    # Native J/B/K/L/etc. do not change replay selections, layer, frame or plots.
    assert snapshots[14][:2] == snapshots[15][:2]
    for snapshot in snapshots[1:15]:
        np.testing.assert_array_equal(snapshot[2], snapshots[0][2])
        np.testing.assert_array_equal(snapshot[3], snapshots[0][3])
    # T toggles transparency exactly once, not once natively and once in Python.
    assert snapshots[14][2][mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
    assert not snapshots[15][2][mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
    assert "PLAYING" in snapshots[16][0] and "frame 101/" in snapshots[17][0]
    assert "PAUSED" in snapshots[18][0]
    assert KEY_HELP in snapshots[0][0]


@pytest.mark.parametrize("key", ["6", "7"])
@pytest.mark.parametrize("play", [False, True])
def test_no_anomaly_navigation_preserves_playback(monkeypatch, key, play):
    model, view = load(V2)
    report = validate_motion(view, model)
    assert not len(report.anomaly_frames)
    writes = []
    set_frame = ReplayState.set_frame

    def record_frame(state, frame):
        writes.append(frame)
        set_frame(state, frame)

    monkeypatch.setattr(ReplayState, "set_frame", record_frame)
    snapshots = run_viewer_keys(
        monkeypatch,
        model,
        {"file": report, "reference": report},
        [[], [ord(key)], [], []],
        view="reference",
        start_frame=0,
        play=play,
        loop=True,
    )
    for i, snapshot in enumerate(snapshots):
        assert f"frame {i if play else 0}/" in snapshot[0]
        assert ("PLAYING" if play else "PAUSED") in snapshot[0]
        if i:
            assert "No anomaly frames detected in current view" in snapshot[0]
            assert "playback unchanged" in snapshot[0]
    # Initialization and normal playback only; no extra state write on 6/7.
    assert writes == (list(range(5)) if play else [0])


@pytest.mark.parametrize("key", ["6", "7"])
def test_no_anomaly_navigation_does_not_interrupt_loop(monkeypatch, key):
    model, view = load(V2)
    report = validate_motion(view, model)
    last = len(view.data.joint_pos) - 1
    snapshots = run_viewer_keys(
        monkeypatch,
        model,
        {"file": report, "reference": report},
        [[], [ord(key)], [], [300]],
        start_frame=last - 1,
        play=True,
        loop=True,
    )
    for snapshot, frame in zip(snapshots[:3], (last - 1, last, 0)):
        assert f"frame {frame}/" in snapshot[0] and "PLAYING" in snapshot[0]
    assert "No anomaly frames detected" in snapshots[2][0]
    # Switching views clears the old notification and still pauses as intended.
    assert "No anomaly frames detected" not in snapshots[3][0]
    assert "PAUSED" in snapshots[3][0]
