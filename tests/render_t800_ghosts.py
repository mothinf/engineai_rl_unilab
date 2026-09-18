"""Manual/offscreen visual acceptance using the production scene and overlays.

Run with MUJOCO_GL=egl and pass an output directory. Writes screenshots and
measurements, never modifies source NPZs. This is not a desktop-GUI smoke test.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from unilab.tasks.motion_tracking.common.motion_loader import MotionData

from engineai_rl_unilab.tasks.t800.motion_npz import STATE_FIELDS
from engineai_rl_unilab.tasks.t800.motion_ghosts import (
    GHOST_NAMES,
    build_ghosts,
    quat_mul,
)
from engineai_rl_unilab.tasks.t800.motion_validation import load_views, validate_motion
from engineai_rl_unilab.tasks.t800.replay import ReplayState, draw_ghosts, ghost_texts


def render(
    model, report, path, enabled, frame=799, width=1500, height=1000, closeup=False
):
    state = ReplayState(model, report.view)
    state.set_frame(frame)
    body = model.body("LINK_ANKLE_ROLL_L").id
    camera = mujoco.MjvCamera()
    camera.lookat[:] = state.data.xpos[state.root]
    camera.distance = 3.5
    camera.azimuth, camera.elevation = 135, -15
    if closeup:
        camera.lookat[:] = state.data.xpos[body]
        camera.lookat[2] += 0.05
        camera.distance = 0.8
        camera.azimuth, camera.elevation = 90, -20
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    with mujoco.Renderer(model, height=height, width=width, max_geom=3000) as renderer:
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        renderer.update_scene(state.data, camera=camera, scene_option=option)
        draw_ghosts(renderer.scene, state, report, enabled, body)
        renderer.render()
        viewport = mujoco.MjrRect(0, 0, width, height)
        for _, location, left, right in ghost_texts(state, report, enabled, body, True):
            assert max(len(left), len(right)) < mujoco.mjMAXOVERLAY
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                location,
                viewport,
                left,
                right,
                renderer._mjr_context,
            )
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, viewport, renderer._mjr_context)
        Image.fromarray(pixels[::-1]).save(path)
        return {
            "frame": frame,
            "scene_geoms": renderer.scene.ngeom,
            "ghosts": sorted(enabled),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    model_path = root / "assets/robots/t800/scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = tuple(
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0
    )
    reports = {}
    for name, suffix in (("old", "mujoco"), ("v2", "mujoco_v2")):
        path = root / f"assets/motions/t800/dance1_subject2_t800_first18s_{suffix}.npz"
        _, views = load_views(path, model_path, names, "mujoco")
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
        reports[name] = build_ghosts(
            validate_motion(views["file"], model), model, present_fields=fields
        )
    for name, field in (
        ("orientation-error", "body_quat_w"),
        ("joint-speed-error", "joint_vel"),
    ):
        view = reports["v2"].view
        view = replace(
            view,
            label=f"INJECTED / {name}",
            data=MotionData(
                **{key: getattr(view.data, key).copy() for key in STATE_FIELDS}
            ),
        )
        if field == "body_quat_w":
            i = view.body_names.index("LINK_ANKLE_ROLL_L")
            delta = np.array([np.cos(0.6), np.sin(0.6), 0, 0])
            view.data.body_quat_w[775:825, i] = quat_mul(
                delta, view.data.body_quat_w[775:825, i]
            )
        else:
            i = view.joint_names.index("J05_ANKLE_ROLL_L")
            view.data.joint_vel[775:825, i] += 4
        reports[name] = build_ghosts(
            validate_motion(view, model),
            model,
            present_fields=reports["v2"].present_fields,
        )
    measurements = {}
    cases = [
        ("old-all", "old", set(GHOST_NAMES), 1500, 1000),
        ("v2-all", "v2", set(GHOST_NAMES), 1500, 1000),
        ("old-body-velocity", "old", {"body-velocity"}, 1500, 1000),
        ("v2-body-velocity", "v2", {"body-velocity"}, 1500, 1000),
        ("orientation-error", "orientation-error", {"pose"}, 1500, 1000),
        ("joint-speed-error", "joint-speed-error", {"joint-velocity"}, 1500, 1000),
        ("old-small", "old", set(GHOST_NAMES), 1000, 700),
    ]
    for filename, key, enabled, width, height in cases:
        path = args.output_dir / f"{filename}.png"
        measurements[filename] = render(
            model, reports[key], path, enabled, width=width, height=height
        )
        print(path)
    for key, layer in (
        ("orientation-error", "pose"),
        ("joint-speed-error", "joint-velocity"),
    ):
        path = args.output_dir / f"{key}-closeup.png"
        measurements[f"{key}-closeup"] = render(
            model, reports[key], path, {layer}, closeup=True
        )
        print(path)
    for key, report in reports.items():
        (args.output_dir / f"{key}-report.json").write_text(
            json.dumps(report.as_dict(), indent=2, allow_nan=False) + "\n"
        )
    (args.output_dir / "renders.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
