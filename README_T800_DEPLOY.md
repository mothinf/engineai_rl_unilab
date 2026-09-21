# T800 真机部署与操作（官方 Native SDK）

执行顺序：**连接机器人 → 下发 SDK → RC02 解锁 → 清理旧进程并启动 → F710 进入 dance → 退出并清理进程**。

本文基于 `0921.md` 的现场记录，并核对了本地已适配 UniLab 的官方 Native SDK。真机由 SDK 的 `src_executor` 控制，dance 使用已准备好的 MNN 策略与 NPZ 轨迹。下面直接使用现有部署包；更换策略的方法放在文末。

| 操作位置 | 本文使用的地址 / 目录 |
| --- | --- |
| 开发机：本仓库 | `/home/pc825/engineai_rl_unilab`（训练与本文档） |
| 开发机：官方 SDK | `/home/pc825/ws/lzy/engineai_robotics_native_sdk`（编译、下发） |
| 真机：板载 PC | `user@192.168.0.163` |
| 真机：部署目录 | `~/projects/uni_t800_ppo` |

地址和目录来自本次使用的设备，换设备时替换为对应值。所有标注“真机”的命令都在 SSH 登录后的终端执行。

## 1. 连接机器人【开发机】

机器人开机，开发机通过网线或机器人 Wi-Fi 接入机器人网络，然后登录：

```bash
ssh user@192.168.0.163
```

首次部署时创建目录，再退出回到开发机：

```bash
mkdir -p ~/projects/uni_t800_ppo
exit
```

## 2. 编译并下发 SDK【开发机】

进入 **SDK 目录**：

```bash
cd /home/pc825/ws/lzy/engineai_robotics_native_sdk
```

首次部署时，将 `install.sh` 的远端参数设为：

```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/uni_t800_ppo"
```

首次编译或修改 C++ 后，在开发机宿主系统执行以下命令；已有对应编译产物、只更新 MNN / NPZ / YAML 时跳过编译：

```bash
./build.sh -a aarch64 -r humble -t release -j2
```

这里使用 aarch64、ROS Humble；若设备使用 Jazzy，将 `-r humble` 改为 `-r jazzy`。

下发程序与资源：

```bash
./install.sh t800 robot aarch64
```

出现密码提示时，输入真机用户 `user` 的 SSH 密码。脚本会同步程序、T800 资源和启动脚本，并设置真机运行模式。

**目标目录必须专用于这套 SDK**：下发使用 `rsync --delete`，会删除目标目录中不在部署包内的文件（当前脚本保留 `logs/`）。若上一轮程序还在运行，先按第 6 步结束，再重新下发。

程序和资源均未更新时，下次使用直接从第 3 步开始。

## 3. 用 RC02 解除开机急停【机器人现场】

1. 使用官方 RC02 按 `BACK + START` 解除开机急停，按官方流程进入**白灯 / 可运行状态**。
2. 将 Logitech F710 接收器插在真机板载 PC 上；手柄设为 **X 模式，MODE 灯灭**。
3. 松开手柄按键、让摇杆回中，再启动下面的 SDK。

当前仍由 RC02 完成开机解锁，F710 负责随后 SDK 内的状态切换。保留实体急停；首次测试使用吊架或辅助支撑，人员远离动作范围。

## 4. 清理旧进程并启动【真机】

开发机打开 SSH：

```bash
ssh user@192.168.0.163
```

在真机上执行。**清理前让机器人停止运动并获得支撑**；旧 SDK 若仍在前台，先在原终端按 `Ctrl+C`，并关闭官方机器人管理服务 / 网页。

```bash
cd ~/projects/uni_t800_ppo

# 停止官方运控服务，再结束残留进程；没有匹配进程时 pkill 无输出。
sudo systemctl stop robotics.service
sudo pkill -TERM -x src_executor
sudo pkill -TERM -f '(^|/)motor_service_server([[:space:]]|$)'

# 两条命令均无输出后，再执行启动命令。
pgrep -af src_executor
pgrep -af motor_service_server
```

若仍有输出，先处理对应旧进程，见文末“进程冲突”；同一时间只能运行一套运控程序。

```bash
sudo ./run_robot.sh t800
```

保持该终端打开。启动后先松键、摇杆回中至少 0.25 秒，再操作 F710。

## 5. 按状态顺序进入 dance【F710】

### 执行阶段状态图

实线为进入 dance 的顺序，虚线为退出或返回路径。`passive` 是阻尼状态，`idle` 是空闲 / 禁用电机控制状态，两者不同。

```mermaid
flowchart LR
    I["idle · 空闲"] -->|"LB + RB"| P["passive · 阻尼"]
    P -->|"LB + A"| S["pd_stand · 站立"]
    S -->|"LB + B"| W["walk · 行走策略"]
    W -->|"RB + B"| D["dance · 动作跟踪"]
    S -.->|"LB + RB"| P
    W -.->|"LB + RB"| P
    D -.->|"LB + RB / 播放完成自动返回"| P
    D -.->|"LB + B"| W
    W -.->|"LB + A"| S
    P -.->|"LB + START"| I
```

| 顺序 | 当前状态 → 目标状态 | F710 按键 | 操作说明 |
| --- | --- | --- | --- |
| ① | `idle → passive` | `LB + RB` | 上阻尼，等待使能成功 |
| ② | `passive → pd_stand` | `LB + A` | 等待站稳 |
| ③ | `pd_stand → walk` | `LB + B` | 先保持摇杆回中，确认稳定 |
| ④ | `walk → dance` | `RB + B` | 开始动作跟踪 |
| ⑤ | `dance → passive` | 无需按键 | 轨迹播放结束自动回到阻尼 |

每次组合键生效后立即松开，再操作下一步。若已经处于 `passive`，从第 ② 步继续。

- **dance 只能从 walk 进入**，不能从 `pd_stand` 直接进入。
- 测试中需要中止动作，按 `LB + RB` 回到 `passive`；软件按键无效时使用实体急停。阻尼状态不会主动维持站立平衡。
- `dance → walk` 也允许用 `LB + B` 手动切回；首次测试中止时优先回 `passive`。
- 本地版本 `walk → dance` 为直接切换，没有额外入场插值，第一帧可能产生冲击。

## 6. 结束实验并清理进程【F710 → 真机】

1. 按 `LB + RB` 回到 `passive`（dance 自然结束时已经在此状态）。
2. 确保机器人已获得支撑，再按 `LB + START` 从 `passive` 进入 `idle`。
3. 在运行 SDK 的 SSH 终端按 `Ctrl+C`，然后执行下面的清理命令。

```bash
# 在真机板载 PC 上执行；结束后也要停服务、清理残留。
sudo systemctl stop robotics.service
sudo pkill -TERM -x src_executor
sudo pkill -TERM -f '(^|/)motor_service_server([[:space:]]|$)'

pgrep -af src_executor
pgrep -af motor_service_server
```

两条 `pgrep` 均无输出后，本轮结束。这里清理的是服务与进程，不是磁盘缓存。

再次启动时重复第 3～5 步；现场曾遇到进入 `idle`、退出并重启 SDK 后无法重新上阻尼的问题，处理方法见下文。

## 常见问题

### 进入 idle 或重启 SDK 后，LB + RB 无法上阻尼

现场记录的现象是：`LB + START → idle` 后退出进程，再次启动 SDK 时无法进入阻尼，并出现 IMU 初始化相关输出。尚不能仅凭该现象确定是 idle、重启、使能还是 IMU 链路导致。

先让机器人获得支撑，停止 SDK 并执行第 6 步的进程清理，再用 RC02 按官方流程恢复白灯 / 可运行状态，重新启动 SDK。启动后松键、摇杆回中，再按一次 `LB + RB`。若仍失败，保留完整启动及使能失败日志排查，不要连续按键尝试进入站立。

F710 的 `LB + RB` 是 SDK 的阻尼 / 使能请求，不能代替 RC02 的 `BACK + START` 解除开机急停。

### 已进入 walk，但左摇杆不能控制行走

先看 F710 的 **MODE 灯**。现场曾因误触 MODE 键，导致左摇杆与十字键功能互换；将 MODE 灯切回熄灭，再在可控条件下重试。

### F710 完全没有反应

接收器应插在**运行 SDK 的真机板载 PC**，手柄使用 X 模式、MODE 灯灭。启动或重连后松开所有动作键，摇杆回中至少 0.25 秒。

在真机检查设备是否存在：

```bash
ls -l /dev/input/js*
ls -l /dev/input/by-id/
```

正常初始化时日志包含：

```text
InputCommandArbiterRunner: hardware source 'gamepad' Init succeeded at index 0
```

旧记录曾采用 `gamepad → rc02` 优先级；当前核对的本地源码已采用 **F710 独占 SDK 输入**，不会自动回退到 RC02。RC02 仍用于开机解锁和原有硬件急停。若启动日志显示 SDK 选中了 `rc02`，可能运行的是旧构建，先停止实验并核对部署版本；不要在运动中拔插接收器。

### 进程冲突：src_executor / motor_service_server 正在运行

按第 4 步停服务并清理。`src_executor` 优先在原终端用 `Ctrl+C` 结束；`motor_service_server` 还在时，关闭官方机器人管理服务 / 网页，再结束残留进程。

`pkill` 的 `-TERM` 是正常终止请求，执行后仍需以 `pgrep` 无输出为准。若进程持续存在或自动重生，先处理启动它的服务，不要叠加启动新的 SDK。

### SSH / 下发 Connection timed out

确认机器人开机、网线连接正常或开发机已接入机器人 Wi-Fi，然后在开发机执行：

```bash
ip route get 192.168.0.163
ssh user@192.168.0.163
```

SSH 恢复后重新执行 `./install.sh t800 robot aarch64`，无需因为网络中断重新编译。

### 编译时 Docker permission denied

在开发机宿主系统执行一次：

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
docker ps
```

若仍无权限，完整注销 Ubuntu 桌面后重新登录，再执行 `docker ps`。只新开终端可能不会刷新组权限；不要用 `sudo ./build.sh` 代替权限修复。

### 启动报 libsrc_app.so 找不到

进入真机部署目录，通过脚本启动：

```bash
cd ~/projects/uni_t800_ppo
sudo ./run_robot.sh t800
```

脚本负责 ROS 环境、资源路径和动态库路径。直接运行 `_install/bin/src_executor` 会缺少这些设置。

### 更换 MNN 权重或 NPZ 轨迹

在开发机 SDK 目录使用现有转换脚本，显式传入文件路径（将 `/path/to/policy.onnx` 换成实际导出文件）：

```bash
cd /home/pc825/ws/lzy/engineai_robotics_native_sdk
MNN_CONVERT=/tmp/MNN-converter/build/MNNConvert \
  ./scripts/convert_unilab_policy_to_mnn.sh \
  /path/to/policy.onnx \
  assets/config/t800/rl_unilab_motion_tracking/policy_new.mnn
```

`MNN_CONVERT` 指向已安装的转换器；若 `MNNConvert` 已在 PATH 中，可省略该环境变量。不要依赖脚本中的历史默认训练路径。

将配套 NPZ 放入同一资源目录，编辑 SDK 的 `assets/config/t800/rl_unilab_motion_tracking/default.yaml`，例如：

```yaml
policy_file: "rl_unilab_motion_tracking/policy_new.mnn"
trajectory_file_npz: "rl_unilab_motion_tracking/motion_new.npz"
```

上述是示例文件名，路径相对于 `assets/config/t800/`。权重、轨迹、观测定义、关节顺序与配置必须配套；本仓库训练任务与某个历史 SDK 构建不一定使用相同观测维度，不能只按文件名直接替换。

只修改 MNN / NPZ / YAML 时无需重新编译 C++，停止旧程序后重新下发并启动即可。若报模型找不到，核对 YAML 中的文件名、大小写与实际文件是否一致。

### NPZ 报 load_the_npy_file: failed fread

旧版 cnpy 读取 Deflate 压缩 NPZ 可能失败。使用未压缩 **ZIP Stored** 格式（NumPy `np.savez`，不是 `np.savez_compressed`）；怀疑传输损坏时比较两端文件的 `sha256sum`。

当前本地 cnpy 对 Unicode 名称数组也有读取限制，数值数组应放在名称元数据之前。仅改为未压缩不会修复关节顺序、body 索引或策略与轨迹不匹配的问题。

### 日志提示 direct switch 或重复请求同一状态

`No transitioner for motion [ walk ] to [ dance ], direct switch.` 表示直接切换，没有额外过渡器，不是报错。

持续出现 `Manual Transition requested ... to [same state]` 时，松开组合键；一次状态切换只需按一次。

### 起身、躺下与回退官方版本

起身 / 躺下不属于进入 dance 的必经步骤，按键如下：

| 当前状态 | 按键 | 动作与完成后的状态 |
| --- | --- | --- |
| `passive` | `START + 十字上` | `supine_to_stance` 起身，完成后自动进入 `walk` |
| `walk` | `START + 十字下` | `stance_to_supine` 躺下，完成后自动进入 `passive` |

回退原官方程序时，先按第 6 步停止本次 SDK 并清理进程，再进入保留的官方部署目录启动（以下为记录中的目录）：

```bash
cd ~/projects/engineai_robotics
sudo ./run_robot.sh t800
```

本文状态图对应本地适配版本；原官方版本的 dance 完成后可能回到 `walk`，不要套用本地版本的自动返回规则。
