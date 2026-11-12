"""Contract tests for the T800 deployment preparation step."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_t800_unilab import CONFIG, check_motion, check_policy  # noqa: E402


def make_policy(path: Path, inputs: int) -> None:
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "weights", "bias"], ["action"])],
        "test-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, inputs])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 25])],
        [
            helper.make_tensor("weights", TensorProto.FLOAT, [inputs, 25], [0.0] * (inputs * 25)),
            helper.make_tensor("bias", TensorProto.FLOAT, [25], [0.0] * 25),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def make_motion(path: Path, profile: dict, *, fps: object = 50, joint_dtype: object = np.float32) -> None:
    frames = 3
    bodies = ["", "LINK_BASE"]
    positions = np.zeros((frames, len(bodies), 3), dtype=np.float32)
    quats = np.zeros((frames, len(bodies), 4), dtype=np.float32)
    quats[..., 0] = 1.0
    np.savez(
        path,
        fps=np.array([fps]),
        joint_names=np.asarray(profile["joint_names"]),
        body_names=np.asarray(bodies),
        joint_pos=np.zeros((frames, 25), dtype=joint_dtype),
        joint_vel=np.zeros((frames, 25), dtype=np.float32),
        body_pos_w=positions,
        body_quat_w=quats,
        body_lin_vel_w=positions,
        body_ang_vel_w=positions,
    )


class TestPrepareT800(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_valid_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            policy, motion = folder / "policy.onnx", folder / "motion.npz"
            make_policy(policy, 134)
            make_motion(motion, self.profile)
            check_policy(policy, self.profile)
            self.assertEqual(check_motion(motion, self.profile), (3, 50))

    def test_140d_policy_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.onnx"
            make_policy(policy, 140)
            with self.assertRaisesRegex(ValueError, "expected"):
                check_policy(policy, self.profile)

    def test_wrong_fps_and_dtype_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            make_motion(motion, self.profile, fps=50.5)
            with self.assertRaisesRegex(ValueError, "fps"):
                check_motion(motion, self.profile)
            make_motion(motion, self.profile, joint_dtype=np.float64)
            with self.assertRaisesRegex(ValueError, "float32"):
                check_motion(motion, self.profile)

    def test_wrong_root_index_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            make_motion(motion, self.profile)
            profile = dict(self.profile, trajectory_body_index=0)
            with self.assertRaisesRegex(ValueError, "LINK_BASE"):
                check_motion(motion, profile)

    def test_wrong_joint_order_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            motion = Path(tmp) / "motion.npz"
            make_motion(motion, self.profile)
            reordered = dict(self.profile, joint_names=list(reversed(self.profile["joint_names"])))
            with self.assertRaisesRegex(ValueError, "joint_names"):
                check_motion(motion, reordered)


if __name__ == "__main__":
    unittest.main()
