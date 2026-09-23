"""Fail-closed T800 motion admission before allocating the training environment.

Uses replay's numerical checks, not its renderer or optional skeleton. Every
clip is checked separately against the nominal model before domain randomization.
No conversion, data repair, permanent PASS cache or hot-path checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

from unilab.assets.hub import resolve_motion_files
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommandCfg

from .motion_npz import STATE_FIELDS
from .motion_validation import Tolerances, load_views, validate_motion


class MotionValidationError(ValueError):
    """Motion admission failed; no training environment may be constructed."""


def preflight_motion(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Return a config pinned to checked paths, leaving the caller's cfg intact."""
    command = cfg.commands.get("motion")
    if not isinstance(command, MotionCommandCfg):
        raise MotionValidationError(
            "T800 motion preflight requires a configured motion command"
        )
    entity = cfg.scene.entities.get(command.entity_name)
    names = getattr(entity, "joint_names", None)
    if isinstance(names, str) or not names or len(names) != 25:
        raise MotionValidationError(
            "T800 motion preflight requires 25 explicit joint names"
        )
    if cfg.scene.fragment_files or cfg.scene.terrain is not None:
        raise MotionValidationError(
            "T800 motion preflight cannot verify composed/terrain scenes; "
            "use a materialized nominal model_file"
        )
    files = command.motion_file
    if not isinstance(files, str) and not isinstance(files, Sequence):
        raise MotionValidationError(
            "motion_file must be a path or a nonempty sequence of paths"
        )
    requested = [files] if isinstance(files, str) else list(files)
    if not requested or any(not isinstance(p, str) or not p.strip() for p in requested):
        raise MotionValidationError("motion_file must contain nonempty string paths")
    if not np.isfinite(cfg.ctrl_dt) or cfg.ctrl_dt <= 0:
        raise MotionValidationError(
            "T800 motion preflight requires a positive finite ctrl_dt"
        )

    model_file = str(Path(cfg.scene.model_file).resolve())
    paths = []
    for path in requested:
        try:
            # Same resolver as the training loader; pin its result for training.
            paths.append(str(Path(resolve_motion_files(path)).resolve()))
        except (ValueError, TypeError, OSError) as exc:
            raise MotionValidationError(
                f"NPZ preflight blocked: {path}\n{exc}"
            ) from exc

    for path in dict.fromkeys(paths):
        try:
            model, views = load_views(path, model_file, tuple(names), "mujoco")
            view = views["file"]
            with np.load(path, allow_pickle=False) as archive:
                extra = set(archive.files) - {
                    *STATE_FIELDS,
                    "fps",
                    "joint_names",
                    "body_names",
                }
            if extra:
                raise ValueError(f"UNVERIFIED unsupported NPZ fields: {sorted(extra)}")
            # MotionCommand advances one reference frame per control step.
            # Unlike the diagnostic player, training does not resample by fps.
            if not np.isclose(view.fps * cfg.ctrl_dt, 1.0, rtol=0, atol=1e-8):
                raise ValueError(
                    f"fps={view.fps} does not match ctrl_dt={cfg.ctrl_dt}; "
                    "training requires one motion frame per control step"
                )
            report = validate_motion(view, model, Tolerances())
            if report.failed or not report.complete:
                raise ValueError(
                    report.text() + "\nRequired contract checks must all PASS; "
                    "short clips/UNVERIFIED data cannot enter training."
                )
        except (ValueError, KeyError, OSError, RuntimeError, BadZipFile) as exc:
            raise MotionValidationError(
                f"NPZ preflight blocked: {path}\nModel: {model_file}\n{exc}\n"
                "Environment creation aborted. Fix the data or explicitly convert "
                "the supported Isaac source with engineai-convert-npz, then recheck."
            ) from exc
        if report.warnings:
            print(
                f"[NPZ preflight] WARNING: {path}\n{report.text()}\n"
                "Contract checks passed; continuing with temporal diagnostic warnings. "
                "No data was repaired. Inspect with engineai-replay.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[NPZ preflight] PASS: {path} "
                f"({len(view.data.joint_pos)} frames, {view.fps} fps)",
                flush=True,
            )

    checked_command = replace(
        command,
        params=replace(
            command.params, motion_file=paths[0] if isinstance(files, str) else paths
        ),
    )
    return replace(
        cfg,
        scene=replace(cfg.scene, model_file=model_file),
        commands={**cfg.commands, "motion": checked_command},
    )
