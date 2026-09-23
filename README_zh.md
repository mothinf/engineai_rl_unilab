<h1 align="center"> engineai_rl_unilab </h1>

<h3 align="center">
基于 UniLab 包分发的 EngineAI 机器人 RL 训练
</h3>

<p align="center">语言：简体中文 | <a href="README.md">English</a></p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

| ![walk](assets/teaser/walk.png) | ![dance1](assets/teaser/dance1.png) | ![dance2](assets/teaser/dance2.png) |
|:-------------------------------:|:-----------------------------------:|:-----------------------------------:|
|              walk               |               dance1                |               dance2                |

`engineai_rl_unilab` 完全构建在 UniLab 的 PyPI 包分发之上，不依赖 UniLab 源码。
训练、回放、runner/learner/collector 全部来自已发布的 `unilab` wheel；本仓库只
携带任务代码（manager terms）、owner 配置和机器人 XML 资产。

特性包括：

* 两个开箱即用的 T800 25-DoF 任务：walk-flat（Manager-Based PPO / SAC，MuJoCo
  后端；SAC 另有 configured-only 的 mjwarp owner）与 motion tracking（PPO，
  MuJoCo 后端）；
* 任务注册缝（seam）：`UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量让 UniLab 的
  `ensure_registries()` 导入本仓库的任务包（含 spawn 子进程）；
* 配置组合缝：Hydra `--config-dir` 把 `src/engineai_rl_unilab/conf/<algo>` 追加进
  对应算法 UniLab 训练脚本的 config search path，外部 `task=<task>/<sim>` owner
  YAML 直接参与组合；conf 树按算法分目录（`conf/ppo/`、`conf/sac/`），与 UniLab
  仓内布局一致；
* 全部 T800 资产（XML + mesh/纹理）随仓库分发，开箱即用；`assets.py` 仅作兜底——
  文件缺失时才从 Hugging Face 数据集
  [`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots)
  补齐（冷路径）。

## 安装

```bash
git clone https://github.com/unilabsim/engineai_rl_unilab.git
cd engineai_rl_unilab
uv sync
```

`unilab==1.1.0` 从生产 PyPI 解析。`uv pip` 用户的等价命令：

```bash
uv pip install "unilab[mujoco,motrix]==1.1.0" "unisim-core>=1.1.3,<1.1.5"
```

## T800 舞蹈 demo

无需训练，使用仓库内配套权重和 NPZ，打开与 G1 demo 同款的 **Motrix 原生实时窗口**：

```bash
uv run demo t800-dance-18s
uv run demo t800-dance
```

首次运行前执行 `uv sync` 安装配套 Motrix 依赖。默认 CPU、16 个机器人（4×4 排列）、从第一帧开始，
动作结束后 reset 并重复，关闭窗口结束。窗口、视角操作和渲染循环直接复用 UniLab 的
Motrix 实现，不是新增 MuJoCo Viewer 或预录视频播放器。18 秒舞蹈使用
`model_14999.pt` 对应策略和 MuJoCo v2 NPZ（150 Hz 物理）；官方舞蹈使用
`model_29999.pt` 对应策略和 `dance_t800.npz`（500 Hz 物理）。两者控制频率均为
50 Hz，actor/critic 为 140/275 维。它们是策略仿真，不是 `engineai-replay` 的摆姿态回放。

```bash
# 指定推理设备，仍打开 Motrix 实时窗口
uv run demo t800-dance-18s --device cuda
# Motrix 无窗口录制；--steps 只用于录制，省略则录一段完整动作
uv run demo t800-dance --render-mode record
# 保留 MuJoCo 录像模式（不是实时窗口）
MUJOCO_GL=egl uv run demo t800-dance-18s --sim mujoco --render-mode record
```

入口沿用 `demo`；UniLab 原有名称（如 `uv run demo dance`）仍转交上游处理。
录像输出为 `assets/checkpoints/t800/dance18s/play_video.mp4` 或
`assets/checkpoints/t800/dance-official/play_video.mp4`。两份策略原来均在 MuJoCo
训练，现在默认在 Motrix 运行，仍执行严格策略契约检查。18 秒 Motrix 配置将求解器
迭代数设为 10（默认 3 会提前终止）；未修改物理步长、PD、NPZ、权重或终止阈值。
两个后端均已通过两轮完整动作测试，这不代表跨后端状态轨迹逐帧相同。
Motrix 使用当前 UniLab 1.1.0 配套的 `motrixsim-core==0.8.2`，需要可用的桌面图形会话。

数量和显示间距分别由 `motrix_demo18s.yaml` / `motrix_demo_official.yaml` 中的
`training.play_env_num`、`training.render_spacing` 控制（当前为 16、2.0）。实时窗口
初始相机使用 Motrix 默认视角；当前依赖仅在录制路径传递 `training.cam_*`，修改这些
YAML 参数不会改变实时窗口初始视角，实时视角可在窗口内操作。

权重副本统一命名为 `model.pt`，保留原始 `run_config.json`；为支持 CPU 加载，仅将
副本的张量存储设备改为 CPU，数值不变。来源和哈希见
`assets/checkpoints/t800/manifest.json`。目前使用本地资源，不自动下载权重。
自选配套权重可用 `--checkpoint /path/to/model.pt`，同目录必须有训练记录，且仍须
通过严格的策略契约检查。旧 CUDA 存储格式的自选权重在纯 CPU 环境需先制作可移植副本。

官方舞蹈原 NPZ 存在速度时间一致性警告，会打印 `WARN` 后继续；结构／模型契约错误
仍会拦截。demo 不修改或修复 NPZ，也不绕过检查。展示时两支舞均关闭初始化扰动、
从头开始；官方舞蹈的原训练配置仍保留随机初始化。

## 训练

在仓库根目录执行（`env.scene.model_file` 相对仓库根解析）：

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco
uv run engineai-train --algo sac --task engineai_t800_walk_flat --sim mujoco
```

短程冒烟（4 个 env、2 次迭代、不回放）：

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

任意 UniLab Hydra override 均可透传（如 `algo.num_envs=512`、
`training.logger=wandb`）。

## 回放

```bash
uv run engineai-eval --algo ppo --task engineai_t800_walk_flat --sim mujoco --load-run -1
uv run engineai-eval --algo sac --task engineai_t800_walk_flat --sim mujoco --load-run -1
```

## T800 motion tracking

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --load-run -1
```

默认 `mujoco.yaml` 是无 DR 的 15000 轮基线。完整的 134 维 WoState DR 配置保留在
`mujoco_wostate_dr.yaml`（40000 轮），通过独立 profile 选择，不能混用两种策略权重：

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco --profile wostate_dr
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --profile wostate_dr --load-run /path/to/dr/model.pt
```

MuJoCo owner 声明 140 维 actor、275 维 critic、25 维动作和逐关节 action scale；
控制频率 50 Hz，每次控制执行 3 个物理子步，episode 上限 500 步，默认 1024 个
环境训练 15000 轮。这是简化基线——不启用 DR events、观测噪声/延迟、动作延迟
或 deadzone——奖励复用 UniLab 的共享 motion terms，不声明与官方 IsaacLab 任务
完全等价。基线保留随机初始化；18 秒 demo 的配套配置将其清零。默认动作 `dance1_subject2_t800_first18s_mujoco_v2.npz` 随仓库放在
`assets/motions/t800/`，无需运行时下载。

任务验证及短程训练：

```bash
uv run --with pytest pytest tests/test_t800_motion_tracking.py -q
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

这些测试验证配置和 reset/step 契约；训练收敛及实机表现需独立评估。

## T800 真机部署

参见 [T800 真机部署与操作（官方 Native SDK）](README_T800_DEPLOY.md)：从连接机器人、下发程序、启动前清理进程，到手柄状态切换、dance 和结束清理；附状态图与常见问题。

## 仓库结构

```text
assets/robots/t800/                 # 机器人 XML + mesh/纹理（全部入 git，开箱即用）
src/engineai_rl_unilab/
├── assets.py                       # 冷路径资产物化（snapshot_download）
├── cli.py                          # engineai-train / engineai-eval：env var + --config-dir 注入
├── conf/
│   ├── ppo/task/engineai_t800_walk_flat/
│   │   ├── base.yaml               # PPO 任务声明（obs/action/command/event/termination）
│   │   └── mujoco.yaml             # PPO MuJoCo owner（task_name、algo 超参、sim2sim 契约字段）
│   └── sac/task/engineai_t800_walk_flat/
│       ├── base.yaml               # SAC 任务声明（含 curriculum、SAC 奖励权重）
│       ├── mujoco.yaml             # SAC MuJoCo owner
│       └── mjwarp.yaml             # SAC mjwarp owner（configured-only，关闭 kp/kd DR）
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── t800/
        ├── __init__.py             # registry.register_env("EngineAIT800WalkFlat", mujoco/mjwarp)
        └── manager_terms.py        # T800 专属 action/reward terms
```

## License

Apache-2.0.
