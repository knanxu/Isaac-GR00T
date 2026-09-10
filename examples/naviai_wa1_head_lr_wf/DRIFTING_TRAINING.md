# WA1 drifting 云端训练命令（执行前供用户确认）

本分支为 `train/drifting-wa1`，仓库为 `https://github.com/knanxu/Isaac-GR00T.git`。
旧 FM 代码目录 `/home/chenlu/Isaac-GR00T` 和旧训练结果保持原样。

## 同步代码

本地在提交检查通过后推送：

```bash
git -C /home/xukainan/Isaac-GR00T-drifting-train push -u fork train/drifting-wa1
```

云端 DNS 目前无法解析 github.com，Git 2.34.1 也不支持 `http.curloptResolve`。
已用下面的临时 localhost CONNECT 通道成功执行只读 `ls-remote`。
通道只允许 github.com:443，Git 仍验证 github.com 的 HTTPS 证书；不修改系统 DNS 或全局 Git 配置。
IP `20.205.243.166` 是 2026-09-10 本地解析并验证可连接的地址，执行前可重新解析确认。

首次克隆仍在本地发起，脚本通过 SSH 标准输入在云端执行：

```bash
ssh -p 20243 chenlu@47.97.38.144 \
  'python3 - --ip 20.205.243.166 -- clone --single-branch --branch train/drifting-wa1 https://github.com/knanxu/Isaac-GR00T.git /home/chenlu/Isaac-GR00T-drifting-train' \
  < /home/xukainan/Isaac-GR00T-drifting-train/scripts/training/github_sync.py
```

已有该独立目录时，在云端通过同一通道 pull：

```bash
python3 /home/chenlu/Isaac-GR00T-drifting-train/scripts/training/github_sync.py \
  --ip 20.205.243.166 -- \
  -C /home/chenlu/Isaac-GR00T-drifting-train pull --ff-only origin train/drifting-wa1
```

同步只取代码，设置 `GIT_LFS_SKIP_SMUDGE=1`，不下载无关的示例视频或平台 wheel。
训练数据、base 权重和依赖使用云端已有文件。
沿用旧环境的 Python 和已安装依赖；每个新进程显式设置新目录的 `PYTHONPATH`，
不执行对旧环境的 editable reinstall、uv sync 或依赖升级。
现有 FM 推理进程保持运行；四卡 smoke 若显存不足则报告失败，不自动停止该进程。

## 检查实际训练参数

以下命令在云端执行（先 `ssh -p 20243 chenlu@47.97.38.144`）：

```bash
DRIFT_REPO=/home/chenlu/Isaac-GR00T-drifting-train
DRIFT_PY=/home/chenlu/Isaac-GR00T/.venv/bin/python
FM_COMMAND=/mnt/datadisk/chenlu/gr00t/runs/express_mix_n17_base_224_20260909/train_command.json
DRIFT_RUNS=/mnt/datadisk/chenlu/gr00t/runs

"$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" \
  --run-dir "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_smoke_20260910" \
  --phase smoke
```

不带 `--execute` 只打印完整 torchrun 命令，不创建目录或启动训练。
35 份训练路径和 5 份验证路径从原命令逐项继承，不重新划分、不重新转换数据。
新参数为：

```text
--action-head-type drifting
--drifting-gen-per-label 4
--drifting-temperatures 0.02 0.05 0.2
--drifting-per-timestep-loss
--drifting-lora-rank 16
--drifting-lora-alpha 32
--drifting-lora-dropout 0.05
--no-tune-visual --no-tune-llm
--tune-projector --tune-diffusion-model
--global-batch-size 8 --gradient-accumulation-steps 8
--state-dropout-prob 0
--learning-rate 1e-4 --warmup-ratio 0.05 --weight-decay 1e-5
--no-save-only-model --no-resume-from-checkpoint --no-use-wandb
```

## 四卡 smoke（20 步）

```bash
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" \
  --run-dir "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_smoke_20260910" \
  --phase smoke --execute \
  > "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_smoke_20260910-launcher.log" 2>&1 < /dev/null &
```

四卡 smoke 使用同一 LoRA、microbatch 和累积配置；每 10 步保存/验证。
训练后重新加载最后一个完整 checkpoint，在原 FM 的三个机器人验证观测上检查
224×224 processor、动作 shape、有限值及每次推理恰好一次动作网络前向。
`train_status.json` 的 `state=complete` 和 `checkpoint_reload_passed=true` 才表示通过。

## 正式训练（20,000 步）

```bash
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" \
  --run-dir "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_20260910" \
  --phase train \
  --smoke-run "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_smoke_20260910" \
  --execute \
  > "$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_20260910-launcher.log" 2>&1 < /dev/null &
```

启动器要求 smoke 成功，且代码提交和原 FM 命令摘要相同。
正式训练重新从已确认的官方 base 开始；不恢复 smoke 或 FM 的 optimizer。
输出目录必须不存在；不会覆盖原实验或自动续训。

日志为新目录中的 `train.log`、`reload.log`，状态为 `train_status.json`。
`train_command.json` 记录实际 argv、环境白名单、源命令 SHA256 和代码提交；
`data_split.json` 原样复制已有划分。原文件不被修改。

最后 checkpoint 预期为：

```text
/mnt/datadisk/chenlu/gr00t/runs/express_mix_n17_drifting_lora16_224_20260910/train/checkpoint-20000
```

该目录保存完整基础权重、未合并的 adapters、动作头和 optimizer 状态。
训练完成和 checkpoint 重载成功不等于实机任务成功，本次不执行 rollout 或 RL。

## 已启动训练的 W&B 曲线

`watch_drifting_wandb.py` 独立读取 `train.log`，补传历史并每 30 秒检查新指标。
可以在上述训练运行中启动，不需要重启训练或改变原 FM/W&B 默认配置：

```bash
DRIFT_RUN="$DRIFT_RUNS/express_mix_n17_drifting_lora16_224_20260910"
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/watch_drifting_wandb.py" \
  --run-dir "$DRIFT_RUN" \
  --entity knanxu-zhejiang-university --project finetune-gr00t-n1d7 \
  > "$DRIFT_RUN/wandb_monitor.log" 2>&1 < /dev/null &
```

服务器需要已有 W&B 登录凭据。若系统 DNS 不可用，可额外传入
`--resolve api.wandb.ai=<已核验的IP>` 和
`--resolve storage.googleapis.com=<已核验的IP>`。
这些地址只用于同步进程的临时 localhost HTTPS 代理；TLS 证书仍正常校验，
系统 DNS、训练进程环境和原 FM 服务不受影响。应在启动时重新核验 IP。

链接和同步进度写入 `wandb_monitor.json`；后台日志为 `wandb_monitor.log`。
曲线横轴为 `train/global_step`，包括 `train/loss`、`train/grad_norm`、
`train/learning_rate` 和 `eval/loss`。训练指标原本每 10 步记录，验证每 1000 步记录；
补传的 wall-clock 时间是上传时间，应按训练步数查看历史。
同步进程重启会接续同一 W&B run；重复启动由文件锁拒绝。
它不上传模型/数据，也不将同步进程的 CPU/GPU 开销伪装成训练系统指标。
