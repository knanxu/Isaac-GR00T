# SPDX-License-Identifier: Apache-2.0
"""Drifting-only LoRA for the N1.7 Qwen backbone.

Inject into the inner modules so GR00T remains a normal PreTrainedModel: its
checkpoint contains the frozen weights, unmerged adapters, and full action head.
No PEFT wrapper or adapter-only save format is imposed on GR00T's Trainer.
"""

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer


def apply_drifting_lora(backbone, config):
    config.validate_action_head()
    if config.action_head_type != "drifting" or config.drifting_lora_rank == 0:
        raise ValueError("Drifting LoRA must be explicitly enabled")
    if getattr(backbone, "_drifting_lora_dropouts", None) is not None:
        raise ValueError("Drifting LoRA is already installed")

    # These names target Qwen3-VL's transformer blocks, excluding patch embedding,
    # visual mergers, embeddings, norms, and the unused language output head.
    scopes = (
        (backbone.model.visual, ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]),
        (
            backbone.model.language_model,
            [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
        ),
    )
    # Fail before mutation if the pinned backbone layout changes.
    for module, targets in scopes:
        names = [name for name, _ in module.named_modules()]
        missing = [target for target in targets if not any(n.endswith(target) for n in names)]
        if missing:
            raise ValueError(f"Qwen backbone is missing LoRA targets: {missing}")

    backbone.requires_grad_(False)
    for module, targets in scopes:
        inject_adapter_in_model(
            LoraConfig(
                r=config.drifting_lora_rank,
                lora_alpha=config.drifting_lora_alpha,
                lora_dropout=config.drifting_lora_dropout,
                target_modules=targets,
                bias="none",
            ),
            module,
        )
    # A tuple avoids registering duplicate module/weight names in state_dict.
    backbone._drifting_lora_dropouts = tuple(
        module.lora_dropout for module in backbone.modules() if isinstance(module, LoraLayer)
    )
    for parameter in backbone.parameters():
        if parameter.requires_grad and config.backbone_trainable_params_fp32:
            parameter.data = parameter.data.float()
    backbone.set_frozen_modules_to_eval_mode()
