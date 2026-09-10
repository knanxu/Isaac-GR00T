# GR00T N1.7 可选 drifting 训练

实现参考 [Isaac-GR00T-drifting-loss](https://github.com/phamtrongthang123/Isaac-GR00T-drifting-loss/tree/ea3e356dcadc5e7993867c71fd7cf05d4b78fb80)。
显式传入 `--action-head-type drifting`，动作网络在 t=0 单次前向直接预测完整 action chunk。
“单步”不改变 action horizon，也不跳过视觉语言 backbone。

## FM 兼容约定

- FM 保持默认；旧训练命令不增加参数。缺少动作头类型的旧 checkpoint 继续按 FM 加载。
- FM 不创建 LoRA 模块，保留原损失、随机采样、可训练参数、数据处理、优化器、scheduler、
  BF16、DeepSpeed、保存和续训流程。FM 的 JSON/YAML/W&B 配置不写入 `action_head_type` 或 `drifting_*` 字段。
- 旧 FM 权重键名、形状、checkpoint 文件布局不变，无需转换。
- drifting 使用独立工作目录和输出目录，不修改旧 FM checkpoint；FM → drifting 是权重初始化，
  不能恢复 FM optimizer/scheduler。Trainer 明确拒绝跨目标 resume。
- 本训练分支包含云端 FM 已使用的 WA1 模态、独立验证集和图像处理设置，不包含新增 rollout/RL 功能。

## 已确认的 WA1 训练参数

用户于 2026-09-10 确认：

| 参数 | 设置 |
| --- | --- |
| 基础权重 | 官方 `nvidia/GR00T-N1.7-3B`，revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| 数据 | 从原 FM `train_command.json` 原样继承 35 份训练集、5 份验证集及采样/增强参数 |
| 输入输出 | 三路 224×224 图像，horizon 40，沿用原 state/action 定义和归一化 |
| Backbone | 视觉和语言 attention/MLP 使用 LoRA；基础权重冻结 |
| LoRA | rank 16，alpha 32；adapter dropout 0.05，沿用参考实现默认值 |
| 动作头 | DiT、state/action encoder、decoder、VL 投影等保持完整参数训练 |
| Drifting | G=4，温度 0.02/0.05/0.2，逐时间步 loss |
| Batch | 4 GPU × 每卡 2 × 梯度累积 8 = 每次更新 64 条观测 |
| 优化 | AdamW，LR 1e-4，cosine，warmup 5%，weight decay 1e-5，20,000 步 |
| State dropout | 新 drifting run 为 0；不修改 FM 默认值 |
| 精度/分布式 | BF16 AMP / 现有 DeepSpeed ZeRO-2 |
| 保存 | 每 1,000 步保存完整训练状态，保留 3 份普通 checkpoint 和验证 loss 最低的模型 |

参考 README 的有效 batch 是 32；这里增加累积次数到 8，保留原 FM 的有效 batch 64。
G 只增加每条观测的生成样本数，不计入数据 batch。
TCP delta 已由转换器计算，模态仍使用 `ABSOLUTE/NON_EEF`，避免二次差分；pinch 仍为连续值。

具体同步和启动命令见 [WA1 云端训练](../examples/naviai_wa1_head_lr_wf/DRIFTING_TRAINING.md)。

## LoRA 实现和 checkpoint

`--drifting-lora-rank 0` 是默认值，不创建 adapters。非零 rank 只允许用于 drifting；
同时传 `--tune-visual` 或 `--tune-llm` 会报错，以免把 LoRA 训练误变成全量骨干训练。

Qwen3-VL 的视觉 targets 为 `attn.qkv`、`attn.proj`、`mlp.linear_fc1`、`mlp.linear_fc2`；
语言 targets 为 attention 的 q/k/v/o 和 MLP 的 gate/up/down。视觉 patch embedding、merger、
embedding、norm 和 lm_head 保持冻结。冻结模块处于 eval 模式，只有 adapter dropout 在训练时启用。

加载普通基础权重时，先按原键名严格加载，再初始化 adapters；重新加载 LoRA checkpoint 时，
先按保存的配置重建 adapters，再加载完整权重。`model.config` 和 `action_head.config` 共享实际配置。

LoRA checkpoint 仍由顶层 GR00T `PreTrainedModel` 和原 Trainer 保存，包含：

- 完整骨干基础权重、**未合并**的 LoRA A/B 权重和动作头；不是 adapter-only 文件。
- `config.json` 中保存 `action_head_type=drifting` 和 LoRA/drifting 参数。
- 普通 `checkpoint-*` 包含原 Trainer/DeepSpeed 的 optimizer、scheduler、RNG、global step。
- processor 和实验配置沿用当前训练回调的目录布局。

不自动 merge LoRA。用于恢复训练的 checkpoint 与未来可能导出的 merged 权重是不同产物。
本次仅交付可训练、可重载的 checkpoint；无需另行寻找原 base 才能还原其模型权重。
加载时仍需现有环境中的 Qwen/processor 配置资源；云端沿用已缓存资源并使用 offline 模式。

新训练中使用 `--base-model-path` 加 `--no-resume-from-checkpoint`。
续训时使用原 drifting 命令、相同 LoRA 配置和输出目录，将该开关替换为
`--resume-from-checkpoint`。普通 checkpoint 才有优化器状态；best checkpoint 用于权重加载。
Trainer 会拒绝 LoRA rank/alpha/dropout 不一致的 resume。

## Lessons Learned 的适配

| 参考经验 | N1.7 的处理 |
| --- | --- |
| from_pretrained 覆盖参数 | 测试实际配置、基础权重初始化和完整 checkpoint 重载，不仅检查 CLI 配置 |
| LoRA 与原地修改冲突 | 用真实小型 Qwen3-VL + PEFT + DiT 做两次更新，检查视觉/语言 adapter 梯度；不使用 detach 掩盖错误 |
| DDP find_unused_parameters=True | 保留 FM 分布式设置；本次采用现有 ZeRO-2，四卡 smoke 检查实际可训练参数和保存路径 |
| G=4 大 batch OOM | 每卡 2，累积 8；四卡 smoke 通过后再正式训练，不能凭 CPU 测试断言显存足够 |
| AsyncVectorEnv 故障 | 属于 LIBERO 仿真评估，不引入本次训练流程 |
| Long 多步任务表现下降 | 训练完成后仍需另行评估累计位姿误差、夹持阶段及完整任务成功率 |

没有添加仅名义上启用的 gradient checkpointing 开关。现有 DiT 的标志字段并不代表执行了重计算；
本次依靠小 microbatch 和 LoRA 控制显存，是否足够由四卡 smoke 判定。

逐帧 drift loss 使用归一化的目标力，不能与 FM MSE 直接比较，也不能单凭 loss 最低决定实机最佳策略。
保存 best-eval-loss 仅提供诊断候选；训练完成检查使用最后一个完整 checkpoint。
未来实机评估需比较 TCP 平移/旋转误差、pinch 误差、chunk 边界和任务成功率。

## 验证

`test_drifting_action_head.py` 覆盖直接动作预测、单次 DiT 调用、mask、FP32 loss、BF16、
旧 checkpoint 加载、CLI 及 Trainer 保存/续训。`test_drifting_lora.py` 使用真实缩小的
Qwen3-VL，检查 FP32/BF16 下两个 DiT 变体的 adapter 梯度、冻结参数、dropout、
完整 safetensors 重载、BF16 推理和带 optimizer 的续训。

`test_drifting_fm_compatibility.py` 加载加入 drift 前的源码，逐元素比较 FM 权重、loss、
梯度、RNG、普通推理和 RTC，并比较 HF JSON、YAML 和 checkpoint 文件字节。
可以通过 `GROOT_FM_BASELINE_DIR` 指定云端 FM 代码的只读快照；未指定时使用本分支的 FM 基线提交。

这些 CPU 检查不代表已经完成四卡训练或实机效果验证。
