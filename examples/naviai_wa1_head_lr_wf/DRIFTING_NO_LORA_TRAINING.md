# WA1 drift：不使用 LoRA、关闭梯度累积

2026-09-14 用户已确认训练范围：直接微调视觉编码器和完整动作头，冻结 LLM，
与原 FM 的训练范围一致；不使用 LoRA，`gradient_accumulation_steps=1`。
用户另已确认 batch 与 FM 一致：global batch=64、4 卡、每卡 16。
以下步数和启动命令供执行前确认。

| 设置 | 本次配置 |
| --- | --- |
| 初始化 | 与上版一致的官方 N1.7 base；重新创建 optimizer |
| 数据 | 从原 FM 命令继承全部 35 份训练路径、5 份验证路径及预处理 |
| 训练范围 | `tune_visual=True`、`tune_projector=True`、`tune_diffusion_model=True`、`tune_llm=False` |
| LoRA | rank 0；checkpoint 无 LoRA 权重，不需要合并 |
| GPU / batch | 4 卡，每卡 16，global batch 64，梯度累积 1，有效 batch 64，与原 FM 一致 |
| drifting | G=4，temperatures=[0.02, 0.05, 0.2]，per-timestep loss |
| 优化 | LR=1e-4，warmup=0.05，weight decay=1e-5，state dropout=0 |
| 正式训练 | 20,000 optimizer steps；每 1,000 步验证、保存；保留 3 个 checkpoint |
| W&B | 独立日志同步，记录 loss、eval loss、grad norm、LR 和实际 batch |

上一版每卡 2、累积 8，有效 batch=64；本版每卡 16、累积 1，有效 batch 仍为 64。
因此相同步数的样本处理量保持一致。此次还冻结了此前通过 LoRA 更新的 LLM，
所以这是按 FM 训练范围进行的无 LoRA 对照，不能把全部效果差异都归因于 LoRA 本身。

参考项目的 Lessons Learned 指出其 G=4 配置每卡 batch=4 已会 OOM；
本项目 N1.7、显存和训练范围不同，因此以本机四卡短测为准，不直接承诺 batch=16 可用。
现有 FM/drift 推理服务保留运行；训练和推理共享 GPU，短测包括这部分显存占用。

## 同步本次训练代码

只推送独立训练分支中的训练启动器、重载检查、W&B 同步及对应测试/文档。
本地 ObservationGUILite 修复、TOPPRA、rollout 和 RL 改动不在此分支。

```bash
git -C /home/xukainan/Isaac-GR00T-drifting-train push fork train/drifting-wa1

ssh -p 20243 chenlu@47.97.38.144 \
  'python3 /home/chenlu/Isaac-GR00T-drifting-train/scripts/training/github_sync.py --ip 20.205.243.166 -- -C /home/chenlu/Isaac-GR00T-drifting-train pull --ff-only origin train/drifting-wa1'
```

该 GitHub IP 于 2026-09-14 在本地重新解析；执行时再次核验。代理只用于指定 HTTPS
连接并保持 TLS 证书校验，不改变系统 DNS 或旧 FM 环境。

## 完整参数预览

以下均在云端执行：

```bash
DRIFT_REPO=/home/chenlu/Isaac-GR00T-drifting-train
DRIFT_PY=/home/chenlu/Isaac-GR00T/.venv/bin/python
FM_COMMAND=/mnt/datadisk/chenlu/gr00t/runs/express_mix_n17_base_224_20260909/train_command.json
DRIFT_RUNS=/mnt/datadisk/chenlu/gr00t/runs
DRIFT_SMOKE="$DRIFT_RUNS/express_mix_n17_drifting_nolora_fmscope_ga1_224_smoke_20260914"
DRIFT_RUN="$DRIFT_RUNS/express_mix_n17_drifting_nolora_fmscope_ga1_224_20260914"

"$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" --run-dir "$DRIFT_RUN" \
  --phase train --recipe full-fm-scope
```

不带 `--execute` 只打印完整命令，不创建训练目录或启动训练。
默认不传 `--recipe` 仍使用先前的 LoRA recipe；FM 启动入口完全不变。

## 四卡短测与正式训练

```bash
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" --run-dir "$DRIFT_SMOKE" \
  --phase smoke --recipe full-fm-scope --execute \
  > "$DRIFT_SMOKE-launcher.log" 2>&1 < /dev/null &
```

短测运行 20 步，并验证最后 checkpoint 可以重新加载、没有 LoRA 权重、实际动作网络只
前向一次、三个历史机器人观测输出有限值且 shape 正确。只有其 `train_status.json` 中
`state=complete` 且 `checkpoint_reload_passed=true` 后，才启动：

```bash
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/train_drifting_wa1.py" \
  --fm-command "$FM_COMMAND" --run-dir "$DRIFT_RUN" \
  --phase train --recipe full-fm-scope --smoke-run "$DRIFT_SMOKE" --execute \
  > "$DRIFT_RUN-launcher.log" 2>&1 < /dev/null &
```

启动器要求短测与正式训练使用同一提交、同一源 FM 命令、同一 recipe 和训练参数。
短测失败时不进入正式训练，也不自动停止已有推理服务或减小 batch。
新训练从官方 base 重新开始，不恢复短测、原 FM 或上版 LoRA 的 optimizer。
输出目录必须不存在；已有 FM 和 drift checkpoint 均不覆盖。

## W&B

正式训练创建 `train.log` 后启动独立监控：

```bash
nohup "$DRIFT_PY" "$DRIFT_REPO/scripts/training/watch_drifting_wandb.py" \
  --run-dir "$DRIFT_RUN" \
  --entity knanxu-zhejiang-university --project finetune-gr00t-n1d7 \
  > "$DRIFT_RUN/wandb_monitor.log" 2>&1 < /dev/null &
```

如果云端 DNS 仍不可用，则附加执行时核验的
`--resolve api.wandb.ai=<IP> --resolve storage.googleapis.com=<IP>`，与上一版使用相同的
进程局部 HTTPS 代理。W&B 新 run 标记为 `no-lora`，不会写入原 LoRA/FM run。
链接写入新目录的 `wandb_monitor.json`。

预期产物：`$DRIFT_RUN/train/checkpoint-20000`，包含模型、processor、统计、optimizer 和
Trainer 状态，可直接用于 drift server；不需要再做 LoRA 合并。

参考：[原项目 README](https://github.com/phamtrongthang123/Isaac-GR00T-drifting-loss#lessons-learned)。
