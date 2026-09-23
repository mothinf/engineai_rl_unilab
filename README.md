<h1 align="center"> engineai_rl_unilab </h1>

<h3 align="center">
EngineAI Robot RL Training on the UniLab Package Distribution
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

| ![walk](assets/teaser/walk.png) | ![dance1](assets/teaser/dance1.png) | ![dance2](assets/teaser/dance2.png) |
|:-------------------------------:|:-----------------------------------:|:-----------------------------------:|
|              walk               |               dance1                |               dance2                |

`engineai_rl_unilab` trains EngineAI robots entirely on UniLab's PyPI distribution —
no UniLab source checkout required. Training, playback, and the
runner/learner/collector stack all come from the published `unilab` wheel; this repo
ships only task code (manager terms), owner configs, and robot XML assets.

Features include:

* Two ready-to-run T800 25-DoF tasks: walk-flat (Manager-Based PPO / SAC on MuJoCo;
  SAC also has a configured-only mjwarp owner) and motion tracking (PPO on MuJoCo);
* Task registration seam: `UNILAB_EXTRA_REGISTRY_PACKAGES` lets UniLab's
  `ensure_registries()` import this repo's task package (including spawn subprocesses);
* Config composition seam: Hydra `--config-dir` appends
  `src/engineai_rl_unilab/conf/<algo>` to the UniLab training script's config search
  path, so external `task=<task>/<sim>` owner YAMLs compose directly — the conf tree
  mirrors UniLab's per-algorithm layout (`conf/ppo/`, `conf/sac/`);
* All T800 assets (XML + meshes/textures) ship in git and work out of the box;
  `assets.py` is a cold-path fallback that pulls missing files from the Hugging Face
  dataset [`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots).

## Installation

```bash
git clone https://github.com/unilabsim/engineai_rl_unilab.git
cd engineai_rl_unilab
uv sync
```

`unilab==1.1.0` resolves from production PyPI. The `uv pip` equivalent:

```bash
uv pip install "unilab[mujoco,motrix]==1.1.0" "unisim-core>=1.1.3,<1.1.5"
```

## T800 dance demos

Run the bundled policies in the **same native Motrix live window as G1 demo**
without training (not NPZ pose replay):

```bash
uv run demo t800-dance-18s
uv run demo t800-dance
```

Run `uv sync` first to install Motrix. Defaults: CPU inference, 16 robots in a 4×4 grid,
deterministic initialization at frame zero, and repeated clips until the window
closes. Rendering and interaction use UniLab's existing Motrix implementation,
not a custom MuJoCo Viewer or a prerecorded video player.
The 18-second demo uses the `model_14999.pt` policy and MuJoCo
v2 motion at 150 Hz physics; the official dance uses `model_29999.pt` and
`dance_t800.npz` at 500 Hz physics. Both use 50 Hz control and 140/275 actor/critic
observations. The original official training profile retains random initialization.

```bash
uv run demo t800-dance-18s --device cuda
uv run demo t800-dance --render-mode record
MUJOCO_GL=egl uv run demo t800-dance-18s --sim mujoco --render-mode record
```

`--steps` applies only to recording, which defaults to one full clip. Live runs
reset at clip end before repeating. Both policies were trained in MuJoCo; the
default Motrix demo preserves strict policy contracts. The 18-second Motrix
owner uses 10 solver iterations (the default 3 terminated early), without
changing physics/control timesteps, PD, weights, NPZ or termination thresholds.
Both backends passed two full clips; this does not imply identical state
trajectories. UniLab 1.1.0 pins Motrix 0.8.2; a working desktop graphics session
is required for the live window. Recorded videos are written to
`assets/checkpoints/t800/dance18s/play_video.mp4` or
`assets/checkpoints/t800/dance-official/play_video.mp4`. The unified `demo` entry
also forwards existing UniLab names (such as `demo dance`) to UniLab.
Bundled `model.pt` files preserve
all checkpoint values with CPU tensor storage for portable loading; original
`run_config.json` copies and provenance/hashes are included. No checkpoint is
downloaded automatically. `--checkpoint /path/to/model.pt` requires an adjacent
training record and a matching policy contract. Custom CUDA-storage checkpoints
need a portable copy before loading on CPU-only systems.

The two `motrix_demo*.yaml` owners set `training.play_env_num=16` and
`training.render_spacing=2.0`. Live playback uses Motrix's default initial view;
the pinned backend forwards `training.cam_*` only for recording, not for the
interactive window. Adjust the live view in the window itself.

The original official motion produces temporal consistency **WARN** diagnostics
and continues. Structural/model errors still block playback. Demos never repair
NPZ data or disable validation.

## Training

Run from the repo root (`env.scene.model_file` resolves relative to it):

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco
uv run engineai-train --algo sac --task engineai_t800_walk_flat --sim mujoco
```

Short smoke run (4 envs, 2 iterations, no playback):

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

Any UniLab Hydra override passes through (e.g. `algo.num_envs=512`,
`training.logger=wandb`).

## Playback

```bash
uv run engineai-eval --algo ppo --task engineai_t800_walk_flat --sim mujoco --load-run -1
uv run engineai-eval --algo sac --task engineai_t800_walk_flat --sim mujoco --load-run -1
```

## T800 Motion Tracking

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --load-run -1
```

`mujoco.yaml` is the no-DR, 15000-iteration baseline. The full 134-dimensional
WoState DR owner is preserved as `mujoco_wostate_dr.yaml` (40000 iterations):

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco --profile wostate_dr
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --profile wostate_dr --load-run /path/to/dr/model.pt
```

Baseline and DR checkpoints have different observation contracts and are not interchangeable.

The MuJoCo owner declares a 140-dim actor, 275-dim critic, 25-dim actions with
per-joint action scales; control runs at 50 Hz with 3 physics substeps, episodes cap
at 500 steps, and training defaults to 15000 iterations over 1024 environments. This
is a simplified baseline — no DR events, observation noise/delay, action delay, or
deadzone — reusing UniLab's shared motion reward terms; it does not claim parity with
the official IsaacLab task. The default motion
`dance1_subject2_t800_first18s_mujoco_v2.npz` ships under `assets/motions/t800/`, so no
runtime download is needed.

Validation and a short training run:

```bash
uv run --with pytest pytest tests/test_t800_motion_tracking.py -q
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

The tests cover config and reset/step contracts; training convergence and on-robot
behavior need separate evaluation.

## T800 real-robot deployment

See the [T800 Native SDK deployment and operation guide (Chinese)](README_T800_DEPLOY.md) for connection, deployment, process cleanup, gamepad state transitions, dance, and troubleshooting.

## Repository Layout

```text
assets/robots/t800/                 # robot XML + meshes/textures (all in git, works out of the box)
src/engineai_rl_unilab/
├── assets.py                       # cold-path asset materialization (snapshot_download)
├── cli.py                          # engineai-train / engineai-eval: env var + --config-dir injection
├── conf/
│   ├── ppo/task/engineai_t800_walk_flat/
│   │   ├── base.yaml               # PPO task declaration (obs/action/command/event/termination)
│   │   └── mujoco.yaml             # PPO MuJoCo owner (task_name, algo hyperparams, sim2sim contract fields)
│   └── sac/task/engineai_t800_walk_flat/
│       ├── base.yaml               # SAC task declaration (curriculum, SAC reward weights)
│       ├── mujoco.yaml             # SAC MuJoCo owner
│       └── mjwarp.yaml             # SAC mjwarp owner (configured-only, kp/kd DR off)
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── t800/
        ├── __init__.py             # registry.register_env("EngineAIT800WalkFlat", mujoco/mjwarp)
        └── manager_terms.py        # T800-specific action/reward terms
```

## License

Apache-2.0.
