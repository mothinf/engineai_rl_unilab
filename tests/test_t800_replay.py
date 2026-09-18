"""Pose replay, actual scene construction, CLI and real keyboard event loop."""

from pathlib import Path
from contextlib import nullcontext
from collections import deque
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from engineai_rl_unilab.tasks.t800.motion_ghosts import (
    GHOST_NAMES,
    build_ghosts,
    quat_mul,
)
from engineai_rl_unilab.tasks.t800.motion_validation import load_views, validate_motion
from engineai_rl_unilab.tasks.t800.replay import (
    COLORS,
    ERROR_COLOR,
    KEY_BINDINGS,
    KEY_HELP,
    ReplayState,
    check_viewer_shortcuts,
    contact_status,
    draw_ghosts,
    enqueue_replay_key,
    ghost_texts,
    main,
    marker_endpoints,
    run_viewer,
    warn_motion_report,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/t800/scene_flat.xml"
DIR = ROOT / "assets/motions/t800"
OLD = DIR / "dance1_subject2_t800_first18s_mujoco.npz"
V2 = DIR / "dance1_subject2_t800_first18s_mujoco_v2.npz"


@pytest.fixture(scope="module")
def loaded():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    reports = {}
    for name, path in (("old", OLD), ("v2", V2)):
        _, views = load_views(path, MODEL, names, "mujoco")
        reports[name] = build_ghosts(validate_motion(views["file"], model), model)
    return model, reports


def test_random_access_pose_without_physics(loaded, monkeypatch):
    model, reports = loaded
    good, bad = reports["v2"], reports["old"]

    def forbidden(*args):
        pytest.fail("Pose replay invoked physics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    state = ReplayState(model, good.view)
    reference = ReplayState(model, bad.view)
    for frame in (799, 0, 25, 799, 899, 0):
        state.set_frame(frame)
        reference.set_frame(frame)
        np.testing.assert_allclose(
            state.data.xpos, good.base.fk.body_pos_w[frame], atol=1e-6
        )
        np.testing.assert_array_equal(state.data.xpos, reference.data.xpos)
        assert state.data.time == pytest.approx(frame / good.view.fps)
    state.set_frame(899)
    assert not state.advance()
    state.set_frame(0)
    assert state.advance() and state.frame == 1
    for frame in (-1, 900):
        with pytest.raises(ValueError):
            state.set_frame(frame)


def test_scene_geometry_error_lines_and_orientation_markers(loaded):
    model, reports = loaded
    body = model.body("LINK_ANKLE_ROLL_L").id
    for key, report in reports.items():
        state = ReplayState(model, report.view)
        state.set_frame(799)
        scene = mujoco.MjvScene(model, maxgeom=1000)
        draw_ghosts(scene, state, report, set(GHOST_NAMES), body)
        geoms = scene.geoms[: scene.ngeom]
        assert 0 < scene.ngeom < 1000
        # Selected-body name is in the overlay, never obscuring the foot itself.
        assert not any(g.label for g in geoms)
        assert not any(g.type == mujoco.mjtGeom.mjGEOM_ARROW for g in geoms)
        red = [g for g in geoms if np.allclose(g.rgba, ERROR_COLOR)]
        assert bool(red) == (key == "old")
        for color in COLORS.values():
            assert any(np.allclose(g.rgba, color) for g in geoms)
        for enabled in (set(), {"pose"}, {"body-velocity"}, {"joint-velocity"}):
            scene.ngeom = 0
            draw_ghosts(scene, state, report, enabled, body)
            for name, color in COLORS.items():
                exists = any(
                    np.allclose(g.rgba, color) for g in scene.geoms[: scene.ngeom]
                )
                assert exists == (name in enabled)
    p = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    twist = np.array([np.cos(0.4), np.sin(0.4), 0.0, 0.0])
    a, b = marker_endpoints(p, q), marker_endpoints(p, quat_mul(twist, q))
    np.testing.assert_allclose(a[0], b[0])  # same bone-axis direction
    assert np.linalg.norm(a[1] - b[1]) > 0.02  # transverse arm reveals roll


def test_cli_reports_defaults_removed_options_and_coverage(tmp_path, capsys):
    output = tmp_path / "report.json"
    args = ["--format", "mujoco", "--check-only", "--report", str(output)]
    assert main([*args, "--npz-file", str(OLD)]) == 1
    warnings = capsys.readouterr().err
    assert "[WARNING] NPZ data consistency check FAILED" in warnings
    assert "body_linear_fd" in warnings and "body-velocity position" in warnings
    assert "frame=" in warnings and "body=LINK_" in warnings
    payload = json.loads(output.read_text())["file"]
    assert (
        payload["ghosts"]["layers"]["body-velocity"]["entities"]["LINK_ANKLE_ROLL_L"][
            "status"
        ]
        == "FAIL"
    )
    assert main([*args, "--npz-file", str(V2)]) == 0
    assert "[WARNING]" not in capsys.readouterr().err
    assert json.loads(output.read_text())["file"]["status"] == "PASS"
    assert main(["--check-only", "--report", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["file"]["status"] == "PASS"
    assert set(payload) == {"file"}
    assert "Selected view: file" in capsys.readouterr().out
    assert main(["--headless", "--steps", "3"]) == 0
    assert "frame=3" in capsys.readouterr().out
    assert (
        main(
            [
                "--format",
                "mujoco",
                "--npz-file",
                str(V2),
                "--headless",
                "--start-frame",
                "899",
                "--loop",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    assert "frame=0" in capsys.readouterr().out
    with np.load(V2) as data:
        extra = {key: data[key] for key in data.files}
    extra["additional_signal"] = np.zeros(900)
    custom = tmp_path / "extra.npz"
    np.savez(custom, **extra)
    assert main([*args, "--npz-file", str(custom)]) == 0
    assert "NPZ verification INCOMPLETE" in capsys.readouterr().err
    payload = json.loads(output.read_text())["file"]
    assert payload["status"] == "PARTIAL"
    assert payload["coverage"]["unsupported_fields"] == ["additional_signal"]
    # A different NPZ fps is accepted independently of the training ctrl_dt.
    extra["fps"] = np.array([25])
    np.savez(custom, **extra)
    assert main([*args, "--npz-file", str(custom)]) in (0, 1)
    for removed in (
        ["--mode", "dynamic"],
        ["--sim-dt", ".01"],
        ["--vector-scale", "1"],
        ["--joint", "x"],
        ["--view", "reference"],
    ):
        with pytest.raises(SystemExit) as error:
            main(removed)
        assert error.value.code == 2
    for invalid in (
        ["--ghost-window", "0"],
        ["--ghost-angle-tol-deg", "nan"],
        ["--speed", "-1"],
    ):
        with pytest.raises(SystemExit) as error:
            main(invalid)
        assert error.value.code == 2


def run_keys(monkeypatch, model, reports, batches, **options):
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
            self.index, self.snapshots = 0, []

        def is_running(self):
            if self.index == len(batches):
                return False
            for key in batches[self.index]:
                if key in native_vis:
                    self.opt.flags[native_vis[key]] ^= 1
                if key in native_rnd:
                    self.user_scn.flags[native_rnd[key]] ^= 1
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
            self.texts = "\n".join(t for _, _, a, b in texts for t in (a, b))

        def sync(self):
            self.snapshots.append(
                (
                    self.texts,
                    self.user_scn.ngeom,
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
        start_frame=100,
        body="LINK_ANKLE_ROLL_L",
        play=False,
        speed=1,
        loop=False,
        ghosts="all",
    )
    vars(args).update(options)
    run_viewer(model, reports, args)
    return viewer.snapshots


def test_key_dispatch_and_native_independence(loaded, monkeypatch):
    model, reports = loaded
    old = reports["old"]
    views = {"file": old}
    batches = [
        [],
        [297],
        [298],
        [299],
        [297, 298, 299],
        [301],
        [300],
        [262],
        [263],
        [55],
        [54],
        [56],
        [ord(k) for k in "BJKLNPRVG1234T"],
        [32],
        [],
        [32],
    ]
    snapshots = run_keys(monkeypatch, model, views, batches)
    assert "pose [off]" in snapshots[1][0]
    assert "body-velocity [off]" in snapshots[2][0]
    assert "joint-velocity [off]" in snapshots[3][0] and snapshots[3][1] == 1
    assert "pose [on]" in snapshots[4][0]
    assert "LINK_FOOT_L" in snapshots[5][0]
    assert snapshots[5][:2] == snapshots[6][:2]  # F11 is no longer a custom key
    assert 300 not in KEY_BINDINGS
    assert "frame 101/" in snapshots[7][0] and "frame 100/" in snapshots[8][0]
    frames = old.anomaly_frames
    following = frames[np.searchsorted(frames, 100, side="right") % len(frames)]
    previous = frames[(np.searchsorted(frames, following) - 1) % len(frames)]
    assert f"frame {following}/" in snapshots[9][0]
    assert f"frame {previous}/" in snapshots[10][0]
    assert "frame 100/" in snapshots[11][0]
    assert snapshots[11][:2] == snapshots[12][:2]
    for snapshot in snapshots[1:12]:
        np.testing.assert_array_equal(snapshot[2], snapshots[0][2])
        np.testing.assert_array_equal(snapshot[3], snapshots[0][3])
    assert snapshots[11][2][mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
    assert not snapshots[12][2][mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
    assert "PLAYING" in snapshots[13][0] and "frame 101/" in snapshots[14][0]
    assert "PAUSED" in snapshots[15][0]
    assert KEY_HELP in snapshots[0][0]


@pytest.mark.parametrize("key", [54, 55])
@pytest.mark.parametrize("play", [False, True])
def test_no_anomalies_preserve_playback_and_loop(loaded, monkeypatch, key, play):
    model, reports = loaded
    report = reports["v2"]
    assert not len(report.anomaly_frames)
    snapshots = run_keys(
        monkeypatch,
        model,
        {"file": report},
        [[], [key], [], [56]],
        play=play,
        start_frame=898,
        loop=True,
    )
    for snapshot, frame in zip(
        snapshots[:3], (898, 899, 0) if play else (898, 898, 898)
    ):
        assert f"frame {frame}/" in snapshot[0]
        assert ("PLAYING" if play else "PAUSED") in snapshot[0]
    assert "No anomaly frames detected" in snapshots[2][0]
    assert "No anomaly frames detected" not in snapshots[3][0]
    assert "PAUSED" in snapshots[3][0]
    if play:
        assert "WARM-UP" in snapshots[2][0]


def test_native_shortcut_guard_and_partial_overlays(loaded, monkeypatch, isaac_npz):
    model, _ = loaded
    check_viewer_shortcuts()
    monkeypatch.setitem(KEY_BINDINGS, ord("J"), "pose")
    with pytest.raises(RuntimeError, match="conflict"):
        check_viewer_shortcuts()
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    _, views = load_views(isaac_npz, MODEL, names, "t800_isaac_v1")
    for view in views.values():
        report = build_ghosts(
            validate_motion(view, model), model, present_fields=("extra",)
        )
        state = ReplayState(model, view)
        for frame, body in ((0, 1), (20, 7)):
            state.set_frame(frame)
            for _, _, left, right in ghost_texts(
                state, report, set(GHOST_NAMES), body, True
            ):
                assert (
                    len(left) < mujoco.mjMAXOVERLAY and len(right) < mujoco.mjMAXOVERLAY
                )
        scene = mujoco.MjvScene(model, maxgeom=1000)
        draw_ghosts(scene, state, report, {"body-velocity"}, 7)
        if view.notes:
            assert (
                "UNVERIFIED"
                in ghost_texts(state, report, set(GHOST_NAMES), 7, True)[1][3]
            )


@pytest.mark.parametrize("key", [ord("C"), ord("F")])
def test_contact_keys_never_enter_replay_queue(monkeypatch, key):
    events = deque()
    enqueue_replay_key(events, key)
    assert not events
    enqueue_replay_key(events, 297)
    assert list(events) == [297]
    monkeypatch.setitem(KEY_BINDINGS, key, "pose")
    with pytest.raises(RuntimeError, match="conflict"):
        check_viewer_shortcuts()


@pytest.mark.parametrize("play", [False, True])
def test_native_contacts_survive_ghost_toggles_and_frame_changes(
    loaded, monkeypatch, play
):
    model, reports = loaded
    snapshots = run_keys(
        monkeypatch,
        model,
        {"file": reports["v2"]},
        [[], [ord("C")], [ord("F")], [297, 298, 299], [], [ord("C"), ord("F")]],
        play=play,
        start_frame=568,
    )
    cp, cf = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
    for snap, expected in zip(
        snapshots, ((0, 0), (1, 0), (1, 1), (1, 1), (1, 1), (0, 0))
    ):
        assert (snap[2][cp], snap[2][cf]) == expected
        assert ("PLAYING" if play else "PAUSED") in snap[0]
    assert "C contacts: ON | F forces: ON" in snapshots[2][0]
    assert (
        "No contacts at this frame" if play else "NOT recorded NPZ forces"
    ) in snapshots[2][0]
    assert "pose [on]" in snapshots[2][0] and "pose [off]" in snapshots[3][0]


def test_native_contact_geometry_coexists_with_ghosts(loaded):
    model, reports = loaded
    report = reports["v2"]
    state = ReplayState(model, report.view)
    state.set_frame(568)  # real recorded pose with three contacts; no injected physics
    assert state.data.ncon == 3
    scene = mujoco.MjvScene(model, maxgeom=2000)
    opt, camera = mujoco.MjvOption(), mujoco.MjvCamera()
    cp, cf = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
    for points, forces in ((0, 0), (1, 0), (0, 1), (1, 1)):
        opt.flags[cp], opt.flags[cf] = points, forces
        mujoco.mjv_updateScene(
            model, state.data, opt, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        contacts = [
            g
            for g in scene.geoms[: scene.ngeom]
            if g.category == mujoco.mjtCatBit.mjCAT_DECOR
        ]
        assert len(contacts) == state.data.ncon * (points + forces)
        assert (
            sum(g.type == mujoco.mjtGeom.mjGEOM_ARROW for g in contacts)
            == state.data.ncon * forces
        )
        native_count = scene.ngeom
        native_positions = np.array([g.pos.copy() for g in scene.geoms[:native_count]])
        draw_ghosts(scene, state, report, set(GHOST_NAMES), 7)
        np.testing.assert_array_equal(
            native_positions, [g.pos for g in scene.geoms[:native_count]]
        )
        assert (opt.flags[cp], opt.flags[cf]) == (points, forces)
    state.set_frame(799)
    assert state.data.ncon == 0
    assert "No contacts at this frame" in contact_status(state, opt)


def test_ghost_only_failures_also_warn(loaded, capsys):
    from engineai_rl_unilab.tasks.t800.motion_ghosts import GhostOptions

    model, reports = loaded
    report = build_ghosts(reports["v2"].base, model, GhostOptions(position_tol=0.0001))
    assert not report.base.failed and report.failed
    warn_motion_report(V2, report)
    warnings = capsys.readouterr().err
    assert "FAILED" in warnings and "body-velocity position" in warnings
    assert "Failed numerical checks" not in warnings


def test_viewer_receives_warning_before_launch(monkeypatch, capsys):
    def viewer(*args):
        assert "[WARNING] NPZ data consistency check FAILED" in capsys.readouterr().err

    monkeypatch.setattr("engineai_rl_unilab.tasks.t800.replay.run_viewer", viewer)
    assert main(["--npz-file", str(OLD), "--play"]) == 1
