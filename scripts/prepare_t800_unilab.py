#!/usr/bin/env python3
"""Validate a 134D T800 UniLab ONNX policy and motion for Native SDK deployment."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "assets/config/t800/rl_unilab_motion_tracking/default.yaml"
OUTPUT = CONFIG.parent
FIELDS = {
    "joint_pos": (25,),
    "joint_vel": (25,),
    "body_pos_w": (3,),
    "body_quat_w": (4,),
    "body_lin_vel_w": (3,),
    "body_ang_vel_w": (3,),
}


def check_motion(path: Path, profile: dict) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as motion:
        required = set(FIELDS) | {"fps", "joint_names", "body_names"}
        missing = required.difference(motion.files)
        if missing:
            raise ValueError(f"motion missing fields: {sorted(missing)}")
        fps = np.asarray(motion["fps"])
        if fps.size != 1 or not np.issubdtype(fps.dtype, np.integer) or \
                int(fps.reshape(-1)[0]) != profile["expected_motion_fps"]:
            raise ValueError("motion fps differs from the 50 Hz policy contract")
        names = [str(value) for value in motion["joint_names"]]
        if names != profile["joint_names"]:
            raise ValueError("joint_names differ from SDK policy order; convert/reorder first")
        body_names = [str(value) for value in motion["body_names"]]
        root = int(profile["trajectory_body_index"])
        if root < 0 or root >= len(body_names) or body_names[root] != "LINK_BASE":
            raise ValueError("trajectory_body_index must identify LINK_BASE")
        frames = None
        for field, tail in FIELDS.items():
            value = motion[field]
            if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                raise ValueError(f"{field} must be finite float32")
            if frames is None:
                frames = value.shape[0]
            if field.startswith("joint_"):
                expected = (frames, *tail)
            else:
                expected = (frames, len(body_names), *tail)
            if value.shape != expected:
                raise ValueError(f"{field} shape {value.shape}, expected {expected}")
        if frames is None or frames < 2:
            raise ValueError("motion requires at least two frames")
        norms = np.linalg.norm(motion["body_quat_w"], axis=-1)
        if not np.all(np.abs(norms - 1.0) <= 0.01):
            raise ValueError("body_quat_w contains non-unit quaternions")
        return frames, int(fps.reshape(-1)[0])


def check_policy(path: Path, profile: dict) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("policy ONNX must expose one input and one output")
    dimensions = []
    for endpoint in (model.graph.input[0], model.graph.output[0]):
        shape = endpoint.type.tensor_type.shape.dim
        dimensions.append(int(shape[-1].dim_value) if shape else 0)
    if dimensions != [profile["expected_observation_dim"], len(profile["joint_names"])]:
        raise ValueError(f"policy dimensions {dimensions}, expected [134, 25]")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="exported 134->25 actor")
    parser.add_argument("--motion", type=Path, required=True, help="MuJoCo-layout T800 NPZ")
    parser.add_argument("--mnn-convert", type=Path, help="path to MNNConvert (or use PATH)")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--replace", action="store_true", help="replace existing deploy assets")
    args = parser.parse_args()

    profile = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    check_policy(args.onnx, profile)
    frames, fps = check_motion(args.motion, profile)
    print(f"[prepare] validated ONNX 134->25 and motion {frames} frames at {fps} Hz")
    print(f"[prepare] ONNX sha256={file_hash(args.onnx)}")
    print(f"[prepare] NPZ  sha256={file_hash(args.motion)}")
    if args.check_only:
        return

    converter = args.mnn_convert or shutil.which("MNNConvert")
    if not converter:
        raise RuntimeError("MNNConvert not found; pass --mnn-convert /path/to/MNNConvert")
    model_target = OUTPUT / "policy.mnn"
    motion_target = OUTPUT / "motion.npz"
    if not args.replace and (model_target.exists() or motion_target.exists()):
        raise FileExistsError("deploy assets already exist; pass --replace to update")
    with tempfile.TemporaryDirectory(prefix=".prepare-t800-", dir=OUTPUT) as temp:
        temp_dir = Path(temp)
        staged_model = temp_dir / "policy.mnn"
        staged_motion = temp_dir / "motion.npz"
        subprocess.run(
            [str(converter), "-f", "ONNX", "--modelFile", str(args.onnx.resolve()),
             "--MNNModel", str(staged_model), "--bizCode", "MNN"],
            check=True,
        )
        if not staged_model.is_file() or staged_model.stat().st_size == 0:
            raise RuntimeError("MNNConvert did not produce a nonempty model")
        shutil.copyfile(args.motion, staged_motion)
        os.replace(staged_motion, motion_target)
        os.replace(staged_model, model_target)
    print(f"[prepare] wrote {model_target} and {motion_target}")
    print("[prepare] next: verify sim2sim and SDK simulation before considering robot install")


if __name__ == "__main__":
    main()
