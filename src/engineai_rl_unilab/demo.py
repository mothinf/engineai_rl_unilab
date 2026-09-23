"""Local, paired T800 policy demos using UniLab's existing PPO eval runtime."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from collections.abc import Sequence

from engineai_rl_unilab.assets import ASSETS_ROOT
from engineai_rl_unilab.cli import _build_command, _ensure_registry_env


@dataclass(frozen=True)
class DemoSpec:
    profile: str
    motion: str
    steps: int
    asset_name: str


DEMO_REGISTRY = {
    "t800-dance-18s": DemoSpec(
        "demo18s", "dance1_subject2_t800_first18s_mujoco_v2.npz", 900, "dance18s"
    ),
    "t800-dance": DemoSpec("demo_official", "dance_t800.npz", 1054, "dance-official"),
}


def build_demo_command(
    name: str,
    *,
    checkpoint: Path | None = None,
    device: str = "cpu",
    sim: str = "motrix",
    render_mode: str = "interactive",
    steps: int | None = None,
) -> list[str]:
    """Resolve only bundled/local resources; never choose the latest training run."""
    spec = DEMO_REGISTRY[name]
    if sim == "mujoco" and render_mode != "record":
        raise ValueError(
            "MuJoCo demos support --render-mode record only; use Motrix for the live window"
        )
    if render_mode == "interactive" and steps is not None:
        raise ValueError(
            "--steps applies only to --render-mode record; close the Motrix window to stop live playback"
        )
    checkpoint = (
        checkpoint
        or ASSETS_ROOT / "checkpoints" / "t800" / spec.asset_name / "model.pt"
    ).resolve()
    motion = ASSETS_ROOT / "motions" / "t800" / spec.motion
    for path in (checkpoint, checkpoint.parent / "run_config.json", motion):
        if not path.is_file():
            raise FileNotFoundError(
                f"Demo {name!r} requires {path}. "
                "Keep the checkpoint and its original run_config.json together. "
                "No automatic checkpoint download is configured."
            )
    record_path = checkpoint.parent / "run_config.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    snapshot = record.get("contract_snapshot") if isinstance(record, dict) else None
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError(f"Demo requires a policy contract_snapshot in {record_path}")
    return _build_command(
        "eval",
        [
            "--algo",
            "ppo",
            "--task",
            "engineai_t800_motion_tracking",
            "--sim",
            sim,
            "--profile",
            spec.profile,
            "--load-run",
            str(checkpoint),
            f"training.device={device}",
            f"training.play_render_mode={render_mode}",
            f"training.play_steps={steps if steps is not None else (spec.steps if render_mode == 'record' else 'null')}",
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    from unilab import demo as upstream_demo

    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve upstream demo names and arguments through the unified entrypoint.
    if argv and argv[0] in upstream_demo.DEMO_REGISTRY:
        from unilab.cli import demo_main

        return demo_main(argv)
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Run a paired T800 policy in the same native Motrix window as G1 demo.",
        epilog="UniLab demos are also supported: "
        + ", ".join(sorted(upstream_demo.DEMO_REGISTRY)),
    )
    parser.add_argument("name", choices=sorted(DEMO_REGISTRY))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim", choices=("motrix", "mujoco"), default="motrix")
    parser.add_argument(
        "--checkpoint", type=Path, help="Requires adjacent run_config.json"
    )
    parser.add_argument(
        "--render-mode",
        choices=("interactive", "record"),
        default="interactive",
        help="Default: native Motrix window; record saves a finite video",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Recording only; default one motion length. Live playback ends on window close",
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    try:
        command = build_demo_command(
            args.name,
            checkpoint=args.checkpoint,
            device=args.device,
            sim=args.sim,
            render_mode=args.render_mode,
            steps=args.steps,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _ensure_registry_env()
    print(f"T800 demo: {args.name} | {args.sim} | device={args.device}", flush=True)
    # Owner asset paths are repo-relative. Do not depend on the caller's cwd.
    env = os.environ.copy()
    if args.device == "cpu":
        # A single small policy is slower with a large default BLAS thread pool.
        # Respect explicit user tuning, and apply this only to the demo child.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    return subprocess.run(
        command, cwd=ASSETS_ROOT.parent, env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
