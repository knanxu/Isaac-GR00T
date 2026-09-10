# Drifting / FM 兼容性验证记录

验证日期：2026-09-10。训练分支：`train/drifting-wa1`。

## 范围

基线提交 `df26a4fc9d99962a6509e90f6186b05e2907551d` 记录云端已有的 WA1
模态、35/5 数据划分支持、224 图像处理和 checkpoint 回调；这些是旧 FM 已使用的设置。
后续提交仅添加 drifting / LoRA 训练、checkpoint 验证、同步启动工具和相关测试/文档。
不包含新 rollout/RL 代码或依赖升级。

另从云端旧目录读取源码快照，在独立本地目录中对照；没有修改旧云端代码或 checkpoint。

## 已通过

- 训练/配置/数据/processor/保存续训相关 CPU 测试 117 项通过（3 项 GPU 测试未执行）。
- 随后增加真实 Qwen3-VL FM 兼容测试并通过，共覆盖 118 个不同的 CPU 测试。
- 8 项 FM 对照测试分别针对已保存的基线提交和云端源码快照通过。
- FM 固定随机种子的参数名称、权重、可训练参数、loss、梯度、RNG、普通动作推理和 RTC
  逐元素一致；覆盖 DiT/AlternateVLDiT 和 FP32/BF16 训练。
- 旧、新 FM 的 HF JSON、filtered JSON、实验 YAML 和小模型 checkpoint 文件逐字节一致。
  旧 checkpoint 用新代码严格重载，无 missing/unexpected/mismatched keys。
- 真实小型 Qwen3-VL 的视觉/语言 LoRA 在两次更新中有有效梯度，冻结骨干权重未改变；
  两种 DiT 和 FP32/BF16 均通过。adapter dropout 与冻结模块 eval 状态分别正确。
- 未合并 LoRA 的完整 safetensors 保存、重载、BF16 推理及 Trainer optimizer 续训通过。
- 原 FM 的实际 `train_command.json` 已用于本地解析 smoke/正式命令；数据路径逐项相同，
  35 份训练集、5 份验证集，新的有效 batch 为 8×8=64。
- `pre-commit run --all-files` 的 Ruff、格式和跨平台 manifest 检查通过。
  本机旧 Git 不支持最新 pre-commit 的 `ls-files --deduplicate`，因此在独立 uv tool 环境
  使用 pre-commit 3.7.1；没有更改 FM Python 环境。
- 云端通过临时 GitHub 连接方式完成只读 `ls-remote`，HTTPS 校验保持开启。

## 复现主要测试

在仓库根目录、已有训练依赖环境中执行：

```bash
OMP_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1 python -m pytest \
  tests/gr00t/model/test_action_head.py \
  tests/gr00t/model/test_drifting_action_head.py \
  tests/gr00t/model/test_drifting_lora.py \
  tests/gr00t/model/test_drifting_fm_compatibility.py \
  tests/gr00t/model/test_offline_processor_loading.py \
  tests/gr00t/model/test_processor_image_configuration.py \
  tests/gr00t/data/test_dataset_factory.py \
  tests/gr00t/data/test_fixed_validation_dataset.py \
  tests/gr00t/configs/ tests/gr00t/experiment/ \
  tests/scripts/test_train_drifting_wa1.py \
  -m 'not gpu' -q --timeout=300
```

对照另一份真实云端源码时，设置 `GROOT_FM_BASELINE_DIR` 指向保留仓库相对路径的只读快照。
不设置时，对照上述基线提交，不需要云端访问或本地临时快照。

## 尚未执行

最终同步和训练命令等待用户确认。四卡 20 步 smoke、完整 20,000 步训练以及大模型 checkpoint
重载尚未执行；CPU 检查不代表多卡可用性或实机任务成功。
启动步骤见 [DRIFTING_TRAINING.md](DRIFTING_TRAINING.md)。
