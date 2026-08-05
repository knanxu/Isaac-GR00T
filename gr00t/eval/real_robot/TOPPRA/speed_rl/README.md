# 独立 TOPPRA Speed-RL

该包为 Kion 双臂 rollout 增加四档共享速度策略，不修改现有
`../eval_toppra_bimanual.py` 和 `../kion_client/`。客户端继承原 TOPPRA 调度、轨迹采样、ROS
观测、`DualPose` 发布和 tracking 日志，并使用独立进程训练 Rainbow-DQN。

速度档为 `0.7、1.0、1.3、1.6`。档位为 `s` 时，每只手臂使用：

```text
velocity     = [1, 1, 1, 3, 3, 3] * s
acceleration = [5, 5, 5, 15, 15, 15] * s^2
safety_margin = 1.0
```

## 两台机器从仓库部署

云端推理机和机器人笔记本应检出同一个提交：

```bash
git clone <YOUR_REPOSITORY_URL> Isaac-GR00T
cd Isaac-GR00T
uv sync --all-extras
```

模型 checkpoint、ROS、Kion 私有消息/服务包和经过认证的速度阈值不属于本仓库，需要分别在
两台机器配置。`pyproject.toml` 已包含 `toppra==0.6.3`，无需再复制单独的 agent 文件。

### 1. 云端启动带 feature 的 server

```bash
cp gr00t/eval/real_robot/TOPPRA/speed_rl/server.env.example speed-rl-server.local.env
# 编辑 MODEL_PATH、EMBODIMENT_TAG、SPEED_RL_MODEL_ID
bash gr00t/eval/real_robot/TOPPRA/speed_rl/launch_server.sh \
  speed-rl-server.local.env
```

`SPEED_RL_MODEL_ID` 是 checkpoint 契约的一部分，训练期间不能更改。server 只在显式启用
`--speed-rl-features` 时导出 N1.7 最后一次去噪、最后一个 DiT block 的 action-token feature；
普通 rollout 仍得到 `(actions, {})`，没有额外传输和 feature 保存开销。

直接开放 ZMQ 端口时，应使用云防火墙限制来源 IP。更推荐 SSH 隧道：云端绑定
`127.0.0.1`，笔记本运行 `ssh -N -L 47866:127.0.0.1:47866 USER@HOST`，客户端地址改为
`127.0.0.1:47866`。

### 2. 笔记本验证 server，不连接机器人

更新后的 server 启动后，以下命令会发送一份合成 Kion observation，并同时检查 action、
feature、候选数、horizon、dtype 和契约：

```bash
.venv/bin/python -m gr00t.eval.real_robot.TOPPRA.speed_rl.probe_server \
  --server-host 127.0.0.1 \
  --server-port 47866 \
  --requests 10 \
  --warmup-requests 2 \
  --task "move parcel onto conveyor belt one by one"
```

成功时输出 JSON 且 `status` 为 `ok`，同时报告端到端 median/P95/max 以及 action/feature
payload 字节数。旧 server 即使能返回 action，因为没有
`get_speed_rl_contract` endpoint，也会被明确拒绝；这表示必须在云端拉取本提交并重启。

### 3. 笔记本配置 ROS/Kion 环境

运行 Speed-RL 的同一个 Python 必须能导入 `rospy`、`sensor_msgs`、`geometry_msgs`、
`zj_humanoid.upperlimb`（或 `upperlimb`）和 `zj_humanoid.hand`（或 `hand`）。先加载 ROS 以及
机器人 catkin workspace，再检查：

```bash
source /opt/ros/noetic/setup.bash
source /path/to/robot_ws/devel/setup.bash
.venv/bin/python -c \
  "from gr00t.eval.real_robot.TOPPRA.kion_client.client import load_ros_types; print(load_ros_types())"
```

如果系统 ROS 的 Python 版本与项目 Python 3.12 不一致，需要为该虚拟环境安装/暴露兼容的
ROS Python 包和生成后的 Kion message modules；代码不能替代厂商 SDK。启动前还应确认所有
必需 topic 持续更新，细节见 `../kion_client/README.md`。

### 4. 启动实机客户端

```bash
cp gr00t/eval/real_robot/TOPPRA/speed_rl/client.env.example speed-rl-client.local.env
# 填入左右臂经硬件认证的六分量 twist 阈值，并按实际路径修改 ROS_SETUP
bash gr00t/eval/real_robot/TOPPRA/speed_rl/launch_client.sh \
  speed-rl-client.local.env
```

feature dim、model id、layer 和 dtype 会从 server 自动发现；CLI 中同名 `--feature-*` 参数仅
作为可选断言，写错会 fail-closed。建议先保持 `DRY_RUN=1`，它仍订阅真实 ROS observation、
请求 action 并运行 TOPPRA，但不配置 Servo 或发布运动命令。链路检查通过后才设为 `0`。

不用配置文件时，也可以直接运行：

```bash
python -m gr00t.eval.real_robot.TOPPRA.speed_rl \
  --server-host 127.0.0.1 \
  --server-port 47866 \
  --inference-mode sync \
  --task "move parcel onto conveyor belt one by one" \
  --left-twist-thresholds VX,VY,VZ,WX,WY,WZ \
  --right-twist-thresholds VX,VY,VZ,WX,WY,WZ
```

原始、无 RL 的实机 rollout 入口保持不变；新实验建议统一从 `../rollout/` 启动：

```bash
python -m gr00t.eval.real_robot.TOPPRA.kion_client \
  --server-host 127.0.0.1 --server-port 47866 \
  --inference-mode sync --dry-run
```

## Episode 操作与训练流程

新部署建议使用统一入口 `../rollout/`。默认 `--control-interface gui`，通过
ObservationGUILite 面板调用 `/gr00t_rollout` ROS 服务；需要保留原终端操作时传
`--control-interface terminal`，两者同时使用则传 `both`。

GUI 按钮和终端命令具有相同语义：

- `start`：安装最新 actor 权重、reset epoch、创建/configure Servo 并启动 80 秒 episode。
- `success` / `failure`：停止消费新轨迹并继续发布最后一个 target。
- `abort`：标注 abort 并关闭 Servo。
- `approve PATH_OR_NOTE`：人工检查当前档 tracking 日志后批准该速度。
- `status`：显示 episode、buffer、速度档、mask、twist stale 和违例锁存状态。
- `quit`：只允许在非 RUNNING 状态退出；运行中必须先 `abort`。

首次使用保持 `PHASE=calibration` 和 `INFERENCE_MODE=sync`。标定按
`0.7 → 1.0 → 1.3 → 1.6`，每档两个有效 episode；每档都必须人工批准。abort 或控制故障会
停止标定，不能自动开放下一档。

标定完成后，把客户端配置改为 `PHASE=online` 再启动。在线阶段只有在 replay 至少包含 256
条 transition 且每档至少 32 条时才允许 `start`；共 22 个 episode。每个 episode 结束后执行
`min(2 * 新增 transition 数, 256)` 次更新，actor 只在下一次 `start` 安装，因此 episode
中权重冻结。最后使用 `PHASE=greedy` 运行五个验收 episode。

只有真正被 TOPPRA 激活执行的 speed decision 才会成为 transition。候选规划失败、错过异步
handoff、episode epoch 已变化或从未激活的轨迹只计入 discarded decision。每个 episode 的
`outcome.json` 保存 action、mask、policy version、feature 统计量和最终奖励，但不保存完整
feature。完整 feature 只存在于 replay checkpoint。

Greedy 验收必须等 22 个在线 episode 全部完成，使用冻结 checkpoint 且 `epsilon=0`。验收
episode 不写 replay、不更新网络、也不覆盖训练 checkpoint；同步与异步验收计数分别持久化。

状态、replay、checkpoint 和 episode 日志默认持久化到：

```text
logs/kion_speed_rl/state/
logs/kion_speed_rl/episodes/
```

不要删除 state 目录，否则标定批准、训练计数和 replay 会丢失。checkpoint 会严格校验 server
feature 契约，避免把旧模型训练出的速度 head 静默用于新模型。每个 episode 的
`outcome.json` 使用原子替换写入；如果 replay、checkpoint 或 phase state 最终化失败，status
会显示 `finalization_error` 并阻止同一进程继续 start。

## 安全与异步门槛

左右臂 twist 阈值没有猜测默认值，必须显式填写。任一分量连续 100 个有效控制帧越限后只
锁存 episode 违例标签，不代替机器人 watchdog、急停、关节/碰撞限制。twist stale 时计数
清零，并屏蔽 `1.3/1.6`。

异步模式必须先完成原始异步 rollout 实机验证，并提供：

```json
{
  "raw_async_rollout_verified": true,
  "speed_rl_async_greedy_verified": true
}
```

第一项只开放异步 greedy；异步在线训练还要求第二项。标定始终是同步模式。所有速度不可行
时不会强行拼接新轨迹。

## 自动检查

```bash
python -m pytest tests/gr00t/eval/test_speed_rl.py -v
python -c \
  "from gr00t.eval.real_robot.TOPPRA.speed_rl.baseline import verify_frozen_baseline; verify_frozen_baseline()"
```

测试覆盖 C51、PER、3-step、mask、checkpoint、99/100 帧违例、feature/action 裁剪对齐、
100/300 ms 假 server 延迟、greedy 不污染 replay、episode 持久化和 `act()` 控制预算。
`task_time` shaping、实际阈值以及硬件认证上限仍必须由实机流程确认，代码不会自行推断。
