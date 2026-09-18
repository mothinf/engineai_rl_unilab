"""Synthetic Isaac-layout fixture; no external repo or binary fixture required."""

from pathlib import Path

import numpy as np
import pytest

from engineai_rl_unilab.tasks.t800.motion_npz import SOURCE_BODY_NAMES, STATE_FIELDS


@pytest.fixture(scope="session")
def isaac_npz(tmp_path_factory):
    # Deliberately reverse joints and use the explicit Isaac body layout.
    # This tests format semantics, not the fidelity of an Isaac simulation.
    path = tmp_path_factory.mktemp("isaac") / "source.npz"
    reference = (
        Path(__file__).resolve().parents[1]
        / "assets/motions/t800/dance1_subject2_t800_first18s_mujoco_v2.npz"
    )
    with np.load(reference, allow_pickle=False) as data:
        body_order = [data["body_names"].tolist().index(n) for n in SOURCE_BODY_NAMES]
        payload = {
            field: data[field][:, ::-1]
            if field.startswith("joint_")
            else data[field][:, body_order]
            for field in STATE_FIELDS
        }
        payload.update(fps=data["fps"], joint_names=data["joint_names"][::-1])
    np.savez(path, **payload)
    return path
