# UniLab T800：模型准备、适配与仿真验收

现场操作以 [T800 真机部署与操作主指南](README_T800_DEPLOY.md)为准。本文集中说明模型兼容性、参数与观测扩展、准备脚本和仿真验收。

本分支以[众擎官方 Native SDK](https://github.com/engineai-robotics/engineai_robotics_native_sdk)的 `335c60e` 为基底，保留 SDK 的构建、MuJoCo 仿真、F710 状态机和真机下发流程；只接入 UniLab T800 motion-tracking 所需的配置、runner 防护和独立 sim2sim。训练代码在训练分支，不需要把两套仓库合并。

## 支持范围

当前只支持已核实的 **134 维观测 → 25 维动作**、50 Hz 控制的 T800 策略：参考关节位置/速度 50 维、参考根姿态 6 维、IMU 角速度 3 维、当前关节位置/速度 50 维、上一动作 25 维。动作是相对于 `default_joint_pos` 的残差，不是直接把 NPZ 的 `joint_pos` 下发到电机。

本仓库训练分支的两个 Motrix demo 权重是 **140 维**，不能直接放进这里。带 `base_lin_vel`、`motion_anchor_pos_b` 的策略还需要验证真机状态估计的坐标和有效性；准备脚本会拒绝 140 维权重。更换任何 134 维权重也必须核对训练时的关节顺序、PD、动作比例、控制周期和参考轨迹，维度一致本身不代表安全。

<a id="compatibility"></a>
## 更换权重与近期实验兼容性

**本分支当前提供的是旧 134 维部署实现，不是所有 T800 策略的通用运行器。** 下表于 2026-11-12 对照实际 ONNX、训练配置和实验 SDK profile 核查。近期实验在另一套持续修改的 Native SDK 上进行；那些实验的通过记录不等于本分支已包含相同功能。本文说明如何接入，未将这些扩展代码或模型加入本分支。

| 策略/实验 | Actor → action | 当前分支能否直接换权重 | 需要对齐或扩展的内容 |
|---|---|---|---|
| 原 UniLab WoState 配置 | 134 → 25 | 仅限观测语义、顺序及控制参数均匹配 | 本文默认流程；核对每个模型的训练配置 |
| 10 月 FlashMimic 134 维旁支 | 134 → 25 | 不能仅凭维度认定兼容 | 四组 action scale；相对关节位置的反馈裁剪差异，见下文 |
| A01/JP3、N05、JP3 resetfix 300k、A12 | 329 → 25 | 不支持，准备脚本会拒绝 | 8 帧参考窗口、前两行旋转 6D、绝对关节位置、重力、观测重排；A12 另有阻尼差异 |
| rethink36 F 组 | 329 → 25 | 不支持；也不能直接套用 A 组 profile | 参考偏移改为 `[0,1,2,4,6,8,12,16]` |
| rethink36 C 组 | 137 → 25 | 不支持 | 当前帧参考 q/dq、前两行旋转 6D、绝对关节位置、重力及对应顺序 |
| rethink36 H 组（含 H04） | 653 → 25 | 不支持 | A 组参考窗口 + 五类实际量各 5 帧历史；C++ 已有历史缓冲框架，但缺少新观测项 |
| 历史 D01 | 446 → 25 | 不能走当前完整流程 | 旧观测项加反馈历史可复用 C++ 框架；准备/独立仿真脚本及模型配置仍需适配 |
| 训练分支 140 维 demo | 140 → 25 | 不支持 | 额外线速度、位置观测及其真机来源、坐标系 |

抽查结果：A12、N05 对应的 A02、JP3 300k、C01、F01、H04 的真实 ONNX 均被当前 `check_policy()` 拒绝；134 维旁支通过该项检查，但这只证明 ONNX 格式和输入/输出维度符合要求。**`--check-only` 不检查训练时的 action scale、PD、观测含义，也不比较 ONNX/MNN 推理结果。**

<a id="adapt-policy"></a>
## 0. 按自己的训练配置接入

### 0.1 观测不变时：先对齐配置

以导出权重对应的 resolved config、观测定义和机器人资产为准，逐项更新 [default.yaml](assets/config/t800/rl_unilab_motion_tracking/default.yaml)，然后执行第 1–3 节。仅更改奖励、训练迭代数、优化器或训练地形，不一定需要改 SDK 代码；是否可复用部署配置由导出模型的输入、动作和控制约定决定。

| 核对项 | 本分支配置/位置 | 如何处理差异 |
|---|---|---|
| 关节名称与顺序 | `joint_names`；NPZ 名称和数组列 | 策略输出、参考、PD、scale、限位必须按同一关节顺序对应；不要只改名称而不重排数据 |
| 默认姿态与动作 | `default_joint_pos`、`action_scale`、`resident_control: false` | 使用 `q_target = q_default + scale * action`，再按训练目标限位裁剪；上一动作观测使用缩放/限幅前的策略输出 |
| PD | `joint_stiffness`、`joint_damping` | 从本模型训练配置提取，不能用另一模型效果较好的 PD 直接替换 |
| 目标限位 | `target_position_lower/upper` | 对齐训练执行器目标范围；它们不等同于并联踝可达域或电机扭矩限制 |
| 观测 | `observation_names/history_lengths` | 逐项核对顺序、单位、坐标系、相对/绝对角、旋转展开方式、历史顺序和初值；新观测见 0.2 |
| 参考动作 | `trajectory_file_npz`、`trajectory_body_index` | 当前要求 LINK_BASE 在索引 1、25 关节、50 Hz；不同布局先离线转换并检查 |
| 控制周期 | [task_motion/default.yaml](assets/config/t800/task_motion/default.yaml) 中 `dance.period: 0.02`，profile 中 `expected_motion_fps: 50` | 参考推进、控制周期和训练保持一致；不能只改 fps 数值来重采样动作 |
| 执行延迟 | 本分支 C++ 直接使用当前动作 | 近期模型训练的 1–3 个物理子步延迟不应照搬到真机再叠加；本分支不读取 `action_execution_delay_steps`，写入该字段不会实现延迟 |

近期 FlashMimic 常用 scale 与旧默认值不同，按关节分组如下。头 scale 等消融实验必须使用各自的值。

| 关节组 | scale |
|---|---:|
| 肩全部轴、肘 pitch | 3.5 |
| 髋 pitch、膝 pitch | 2.7 |
| 髋 roll/yaw、腰 yaw、肘 yaw | 1.3 |
| 踝 pitch/roll、头 pitch/yaw | 0.85 |

对应当前 `J00`–`J24` 顺序的完整数组为：

```yaml
action_scale: [2.7, 1.3, 1.3, 2.7, 0.85, 0.85, 2.7, 1.3, 1.3, 2.7, 0.85, 0.85, 1.3, 3.5, 3.5, 3.5, 3.5, 1.3, 3.5, 3.5, 3.5, 3.5, 1.3, 0.85, 0.85]
```

A12 与 JP3/N05 的 Kp 相同：髋 pitch/膝 180，髋 roll/yaw/腰 100，踝 40，上肢/头 50。**A12 的踝 Kd=3、上肢/头 Kd=1；JP3/N05/JP3 300k 分别为 2 和 0.3。** 其余 Kd 为髋 pitch/膝 5、髋 roll/yaw/腰 3。不能把 JP3 300k 的 profile 当作 A12 的配套配置。

对 134 维旁支还有一个容易漏掉的差异：当前 C++ `joint_pos` 会先按 SDK 关节限位裁剪反馈，再减默认姿态；FlashMimic 的 `joint_pos_relative` 使用未裁剪反馈减默认姿态。独立 Python 脚本使用后者。接入该旁支时，应为 C++ 增加匹配的观测项，并确认参考 q/dq 的处理与训练一致；不要直接把不同名字视为完全等价。

### 0.2 观测改变时：配置与代码一起适配

**只将 `expected_observation_dim: 134` 改成 329/653，或粘贴其他 SDK 的 YAML，不能完成适配。** 当前 C++ registry 没有 `ref_root_ori_future_b`、`ref_joint_pos_future`、`joint_pos_absolute`、`joint_pos_relative`、`reference_joint_state`；未知项会被拒绝，参数加载器也没有 `reference_frame_offsets`。

以 A12/JP3/N05 的 329 维为例，需要实现以下约定：

```text
last_action(25) + ref_root_ori_future_b(8×6) + ref_joint_pos_future(8×25)
  + base_ang_vel(3) + projected_gravity(3) + joint_pos_absolute(25) + joint_vel(25)
reference_frame_offsets = [-8,-4,-2,0,1,2,3,4]  # 50 Hz；越界取首/末帧
```

旋转 6D 使用相对旋转矩阵的**前两行按行展开**，旧 134 维使用前两列按行展开，两者不能混用。实际 q 使用绝对角，需核对训练校准偏置与实际编码器校准方式；不能沿用 `q-default`。参考窗口中的过去帧来自动作文件，不是机器人的反馈历史。

C 组为 `last_action25 + 当前参考朝向6 + 当前参考q/dq50 + 角速度3 + 重力3 + 绝对q25 + dq25 = 137`，朝向仍取前两行。F 组沿用 329 维排列，但参考偏移不同。H 组保留 A 的 248 维参考窗口，将 `actions/base_ang_vel/projected_gravity/joint_pos_absolute/joint_vel` 各堆叠 5 帧：按观测项分组、旧到新、首帧重复填充，合计 `248 + 5×81 = 653`。

| 修改入口 | 必须同步处理的内容 |
|---|---|
| [wbt_obs_registry.cc](src/runner/rl_dance_example/src/wbt_obs_registry.cc) 和 [wbt_obs_registry.h](src/runner/rl_dance_example/include/rl_dance_example/wbt_obs_registry.h) | 新观测计算、窗口帧索引、6D 编码；维度函数也按窗口长度计算 |
| [rl_dance_example_param.h](src/data/_param/rl_dance_example_param/rl_dance_example_param.h) | 声明并读取新增参数，例如 `reference_frame_offsets` |
| [rl_dance_example_runner.cc](src/runner/rl_dance_example/src/rl_dance_example_runner.cc) | 将参数传给观测上下文；同步总维度和缓冲分配，保留首帧初始化、原始 action 历史及异常检查 |
| [prepare_t800_unilab.py](scripts/prepare_t800_unilab.py) | 当前只读取固定 `default.yaml`、输出固定文件名；扩展为显式选择 profile，并验证模型与其配套约定。不要仅放宽维度检查 |
| [sim2sim_unilab_t800.py](scripts/sim2sim_unilab_t800.py) | `build_observation()`、134 维门槛及历史状态必须同时扩展；现有 `--config` 只替换参数，不自动实现新观测 |

历史 D01 的反馈历史可以利用现有 C++ 缓冲机制；新 H 组也可复用该机制，但仍需实现新增观测项。引用其他 SDK 的实现时，应按上述入口迁移并重新编译；来源 SDK 的旧构建产物不能代替本分支源码验收。

### 0.3 多模型切换与验收

为每个实验保留独立 YAML、模型、参考动作以及对应训练配置/哈希。单模型且观测未变时，先更新默认 profile 即可。多模型时可复制为 `assets/config/t800/rl_unilab_motion_tracking/my_policy.yaml`；在 [mode.yaml](assets/config/t800/mode.yaml) 的 `mode.sim` 和 `mode.robot` 中，分别将 **`tag: unilab_motion_tracking`** 的 scope 指向 `rl_unilab_motion_tracking/my_policy`。`dance` 的 `param_tag` 仍为 `unilab_motion_tracking`，修改 `rl_dance_example` 的映射不会选中这里的策略。

`policy_file` 和 `trajectory_file_npz` 相对于 `assets/config/t800/`，不是 YAML 所在目录。现有准备脚本没有 `--config`，仍写默认目录；采用独立文件名时，需要显式整理输出并核对 profile 路径，或先按 0.2 扩展准备脚本。Python 仿真可用 `--config .../my_policy.yaml`，同时显式指定配套 `--policy`、`--motion`；目前仅限原 134 维观测实现。

完成接入后按以下顺序验收：

1. **同输入核对观测与目标。** 用同一组参考帧、q/dq、姿态、角速度和上一动作，对比训练端、Python、C++ 的逐项观测与 `q_default + scale * action`。含窗口/历史时覆盖动作首尾、首帧填充、至少 5 个连续控制步和重新进入 dance。
2. **核对推理。** 对真实记录的观测比较 ONNX 和目标平台 MNN 的输出；转换成功或输出维度相同不能代替数值比较。
3. **完整 SDK 仿真。** 除独立 Python 仿真外，运行 C++ executor + SDK MuJoCo，检查关节跟踪、目标限幅、非有限值、进入/退出及参考结束。近期实验中的 direct 关节 PD、日志和其他 SDK 修改不保证已在本分支复现，应记录实际执行模式再比较结果。
4. **记录验证范围。** 区分“配置检查通过”“独立仿真通过”“SDK 仿真通过”和“真机验证”；训练环境成绩与其他 SDK 的真机记录不能直接沿用为本分支结论。


## 1. 准备模型与轨迹（开发机）

在此 deploy 分支根目录操作。先安装 Python 检查/仿真依赖，例如在已有 UniLab Python 环境中使用 `numpy`、`PyYAML`、`onnx`、`onnxruntime`、`mujoco`。另需可执行的 `MNNConvert`；SDK 运行时读取 MNN，而 sim2sim 直接读取同一 ONNX。

以下命令用于已按第 0 节对齐的原 134 维 profile；329/653 等新观测须先完成代码适配。从训练环境导出 actor ONNX 后，先只读校验：

```bash
python scripts/prepare_t800_unilab.py \
  --onnx /path/to/policy.onnx \
  --motion /path/to/dance1_subject2_t800_first18s_mujoco_v2.npz \
  --check-only
```

该脚本要求 NPZ 含六个运动字段、`fps`、`joint_names`、`body_names`，且为 50 Hz、float32、25 关节、根 body 在配置指定的索引 1。它拒绝不匹配的 ONNX 输入/输出维度；不会修改原始文件。这里要求的是 **MuJoCo 布局** NPZ；Isaac 源文件须先在训练仓库离线转换并校验。

为减少 SDK cnpy 与 NumPy 压缩/ZIP64 格式的兼容问题，部署副本建议使用 `np.savez` 保存为不压缩 NPZ，并将 `fps` 明确存为 `int32`。当前 Python 检查允许整数 fps，但 C++ 读取要求 32 位整数；准备脚本只复制文件，不会自动转换存储格式。以下只改容器格式和 fps 整数类型，不转换坐标、关节顺序或帧率，也不覆盖输入文件：

```bash
python - /path/to/source_motion.npz /path/to/reference.npz <<'PY'
from pathlib import Path
import sys
import numpy as np

source, target = map(Path, sys.argv[1:])
with np.load(source, allow_pickle=False) as data:
    arrays = {name: data[name].copy() for name in data.files}
fps = np.asarray(arrays["fps"])
if fps.size != 1 or not np.issubdtype(fps.dtype, np.integer) or int(fps.item()) != 50:
    raise ValueError("Expected an existing 50 Hz motion; this is not a resampler")
arrays["fps"] = fps.astype(np.int32)
with target.open("xb") as output:
    np.savez(output, **arrays)
PY
```

对生成的 `reference.npz` 重新运行上面的 `--check-only`。确认检查通过，再指定转换器（这里使用该部署副本）：

```bash
python scripts/prepare_t800_unilab.py \
  --onnx /path/to/policy.onnx \
  --motion /path/to/reference.npz \
  --mnn-convert /path/to/MNNConvert
```

输出到 `assets/config/t800/rl_unilab_motion_tracking/policy.mnn` 和 `motion.npz`；文件已存在时须显式加 `--replace`。这两个用户数据文件已被 Git 忽略，不随分支分发。默认 YAML 位于 `assets/config/t800/rl_unilab_motion_tracking/default.yaml`，换策略时必须对齐其中的 PD、动作比例和限位。

## 2. 仿真验收（开发机）

先跑独立的 MuJoCo + ONNX sim2sim，不连接电机，也不逐帧覆盖姿态：

```bash
python scripts/sim2sim_unilab_t800.py \
  --policy /path/to/policy.onnx \
  --motion /path/to/reference.npz \
  --headless --steps 900
```

去掉 `--headless` 可打开 MuJoCo 窗口；加 `--realtime` 以真实控制周期播放。该脚本和 C++ runner 共用上述 YAML 的关节顺序、默认姿态、PD、动作比例、限位及根索引。检查动作是否稳定、目标角是否频繁触限、脚底接触和姿态是否合理；“脚本不报错”不是物理或真机安全结论。

然后按官方 SDK 的容器说明编译并运行完整 SDK 仿真：

```bash
./build.sh -j2 -t release
./scripts/build_mujoco.sh
./scripts/run_mujoco.sh t800
```

另开终端，在同一环境、同一分支根目录启动 C++ executor：

```bash
./run.sh t800
```

仅启动 MuJoCo 窗口没有运行策略。仿真中使用 F710 或 [虚拟手柄](tools/virtual_gamepad/README.md)进入相应状态。

`mode.yaml` 默认 `active_mode: sim`。SDK 仿真需要本机 ROS 2 Humble/官方容器、MNN 运行库与 MuJoCo；若环境缺失，独立 Python sim2sim 通过不能代替完整 SDK 构建验收。

## 3. 真机下发与现场操作

完成模型准备与仿真验收后，按 [现场操作主指南](README_T800_DEPLOY.md)执行连接、编译下发、启动前后进程清理和 F710 状态切换。

本分支统一采用 `idle → passive → pd_stand → walk → dance` 的入场顺序，**只能从 `walk` 进入 `dance`**；`pd_stand → dance` 入口已移除。dance 完成后自动返回 passive。当前没有入场轨迹插值，首帧姿态匹配与退出时的支撑要求见主指南。

## 验证边界

准备脚本检查六个运动字段、名称和帧率；部署 runner 再对其实际消费的关节参考、根姿态、观测和动作做运行时检查，并对关节目标角限幅。两者都不能验证机械可达性、扭矩/温度/通讯链路和真机状态估计质量。本分支没有自动 SSH、没有自动下发、没有执行真机试验。

2026-11-12 的兼容性检查覆盖本分支源码、近期实验 profile 和实际 ONNX 格式/维度；该兼容性核查未迁移新观测实现、重跑 SDK 仿真或执行真机试验；现场指南迁入时另将状态配置收紧为仅允许 walk 进入 dance。
