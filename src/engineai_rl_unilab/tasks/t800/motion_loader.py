"""UniLab frame-access interface backed by in-memory T800 adaptation."""

from collections.abc import Sequence

import numpy as np

from unilab.assets.hub import resolve_motion_files
from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader

from .motion_adapter import STATE_FIELDS, adapt_t800_isaac_motion


class T800MotionLoader(MotionLoader):
    """Reuse sampling/gather methods without the base class's NPZ reader."""

    def __init__(
        self,
        motion_file: str | Sequence[str],
        *,
        model_file: str,
        joint_names: Sequence[str],
        body_indices: np.ndarray | None = None,
    ):
        self.motion_files = self._normalize_motion_files(
            resolve_motion_files(motion_file)
        )
        clips = [
            adapt_t800_isaac_motion(
                path, model_file=model_file, joint_names=joint_names
            )
            for path in self.motion_files
        ]
        self.fps = clips[0].fps
        self.joint_names = clips[0].joint_names
        all_body_names = clips[0].body_names
        if any(c.fps != self.fps for c in clips):
            raise ValueError("T800 motion clips must share the same fps")
        indices = (
            np.arange(len(all_body_names))
            if body_indices is None
            else np.asarray(body_indices)
        )
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= len(all_body_names))
        ):
            raise ValueError("body_indices must select valid target MuJoCo body IDs")
        self.body_names = tuple(all_body_names[i] for i in indices)
        self.num_joints = len(self.joint_names)
        self.num_bodies = len(self.body_names)
        self.clip_lengths = np.asarray(
            [len(c.data.joint_pos) for c in clips], dtype=np.int32
        )
        self.num_clips = len(clips)
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.num_frames = int(self.clip_lengths.sum())
        for field in STATE_FIELDS:
            values = [getattr(c.data, field) for c in clips]
            if field.startswith("body_"):
                values = [value[:, indices] for value in values]
            setattr(self, field, np.concatenate(values, axis=0))
