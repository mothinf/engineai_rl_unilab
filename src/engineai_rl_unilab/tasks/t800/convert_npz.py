"""Offline conversion of the verified T800 Isaac export to a MuJoCo NPZ.

Example::

    engineai-convert-npz --input isaac.npz --output motion_mujoco.npz

Input contract: t800_isaac_v1, 25 named hinge joints, 30 source bodies without
world, wxyz quaternions, world-frame root-origin linear/angular velocities.
This is NOT a generic IsaacLab/COM converter. Non-root body states are rebuilt
from joints and root state using the target model, not preserved or repaired.
Output: float32 states, model-order joints, model-order body origins including
world, world-frame velocities, fps and explicit joint_names/body_names.
No resampling, differentiation, physics stepping or joint-limit clipping.
Training uses UniLab's unchanged MotionLoader on the resulting file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from engineai_rl_unilab.assets import ASSETS_ROOT, ensure_t800_assets

from .motion_npz import (
    SOURCE_BODY_NAMES,
    STATE_FIELDS,
    load_t800_isaac_source,
    tracking_fk,
)


def convert_npz(
    input_file: str | Path, output_file: str | Path, *, model_file: str | Path
) -> Path:
    """Write a new MuJoCo NPZ; never overwrite an input or existing output."""
    source, destination = Path(input_file).resolve(), Path(output_file).resolve()
    if source == destination:
        raise ValueError("Input and output must be different files")
    if destination.suffix.lower() != ".npz":
        raise ValueError("Output must have the .npz extension")
    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}; choose a new path"
        )
    with np.load(source, allow_pickle=False) as archive:
        unknown = set(archive.files) - {
            *STATE_FIELDS,
            "fps",
            "joint_names",
            "body_names",
        }
    if unknown:
        raise ValueError(
            f"Unsupported source fields (will not silently discard): {sorted(unknown)}"
        )
    fps, source_names, arrays = load_t800_isaac_source(source)
    model = mujoco.MjModel.from_xml_path(str(Path(model_file).resolve()))
    target_names = tuple(
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    )
    if len(target_names) != 25 or set(target_names) != set(source_names):
        raise ValueError("Target must have the same 25 named T800 joints as the source")
    bodies = ("", *(model.body(i).name for i in range(1, model.nbody)))
    if model.nbody != 31 or set(bodies[1:]) != set(SOURCE_BODY_NAMES):
        raise ValueError("Target must contain the 30 named T800 bodies plus world")
    order = [source_names.index(name) for name in target_names]
    for name in ("joint_pos", "joint_vel"):
        arrays[name] = arrays[name][:, order]
    result = tracking_fk(model, target_names, arrays)
    payload = {name: getattr(result, name) for name in STATE_FIELDS}
    payload.update(
        fps=np.array([fps], dtype=np.int32),
        joint_names=np.asarray(target_names),
        body_names=np.asarray(bodies),
    )
    # Exclusive creation also prevents a competing process's output being lost.
    # If writing fails, remove only the incomplete file created by this call.
    with destination.open("xb") as stream:
        try:
            np.savez_compressed(stream, **payload)
        except BaseException:
            destination.unlink()
            raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument(
        "--source-format", choices=("t800_isaac_v1",), default="t800_isaac_v1"
    )
    args = parser.parse_args(argv)
    try:
        if args.model_file is None:
            ensure_t800_assets()
        path = convert_npz(
            args.input,
            args.output,
            model_file=args.model_file or ASSETS_ROOT / "robots/t800/scene_flat.xml",
        )
        with np.load(path, allow_pickle=False) as archive:
            frames = len(archive["joint_pos"])
            fps = int(archive["fps"][0])
        print(
            f"Saved: {path}\nFormat: MuJoCo body origins / world velocities\nFrames: {frames}; fps: {fps}"
        )
        print(
            "Preserved: joint states (reordered), root pose/velocity and fps.\n"
            "Rebuilt: all non-root body poses/velocities using MuJoCo FK.\n"
            "Input unchanged. Conversion does NOT certify source non-root fields.\n"
            f"Check: engineai-replay --format mujoco --npz-file {str(path)!r} --check-only"
        )
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Conversion error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
