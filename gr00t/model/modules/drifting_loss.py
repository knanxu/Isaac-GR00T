# SPDX-License-Identifier: Apache-2.0
"""Conditional drifting loss with padding excluded from distances and forces.

Adapted from phamtrongthang123/Isaac-GR00T-drifting-loss, commit
ea3e356dcadc5e7993867c71fd7cf05d4b78fb80, gr00t/model/drifting/drifting_util.py
(Apache-2.0). Uses the same detached, multi-temperature attraction/repulsion
target, with generated samples as negatives and one demonstration as positive.
"""

import torch
import torch.nn.functional as F


def drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    action_mask: torch.Tensor,
    temperatures: list[float],
) -> torch.Tensor:
    """Return masked squared errors [B, D], averaged over G generated samples.

    Inputs are generated [B, G, D], positive [B, 1, D], and a binary mask [B, D].
    Pairwise comparisons stay within each observation. Batch statistics exclude
    fully padded observations; feature/force scales exclude padded coordinates.
    With an all-ones mask this matches the reference loss (no fixed negatives).
    Target construction and regression run in FP32, including under BF16 AMP.
    """
    with torch.autocast(device_type=generated.device.type, enabled=False):
        mask = action_mask.bool()
        gen = torch.where(mask[:, None], generated.float(), 0.0)
        pos = torch.where(mask[:, None], positive.float(), 0.0)
        num_generated = gen.shape[1]

        with torch.no_grad():
            old_gen = gen.detach()
            targets = torch.cat((old_gen, pos), dim=1)
            # Direct distances avoid cancellation for nearly identical samples.
            dist = torch.cdist(old_gen, targets, compute_mode="donot_use_mm_for_euclid_dist")
            dist = dist.clamp_min(1e-4)
            valid = mask.any(dim=-1)
            scale = (dist.mean(dim=(1, 2)) * valid).sum() / valid.sum().clamp_min(1)
            dimensions = mask.sum(dim=-1).clamp_min(1).float()
            scale_inputs = (scale / dimensions.sqrt()).clamp_min(1e-3)[:, None, None]
            old_gen_scaled = old_gen / scale_inputs
            targets_scaled = targets / scale_inputs
            dist_normalized = dist / scale.clamp_min(1e-3)
            diagonal = torch.eye(num_generated, device=gen.device)
            dist_normalized = dist_normalized + F.pad(diagonal, (0, pos.shape[1])) * 100.0

            force = torch.zeros_like(old_gen_scaled)
            for temperature in temperatures:
                logits = -dist_normalized / temperature
                affinity = (logits.softmax(dim=-1) * logits.softmax(dim=-2)).clamp_min(1e-6).sqrt()
                negative_affinity = affinity[:, :, :num_generated]
                positive_affinity = affinity[:, :, num_generated:]
                coefficients = torch.cat(
                    (
                        -negative_affinity * positive_affinity.sum(dim=-1, keepdim=True),
                        positive_affinity * negative_affinity.sum(dim=-1, keepdim=True),
                    ),
                    dim=-1,
                )
                force_at_temperature = coefficients @ targets_scaled
                force_at_temperature -= coefficients.sum(dim=-1, keepdim=True) * old_gen_scaled
                force_norm = force_at_temperature.square().sum() / (
                    num_generated * mask.sum()
                ).clamp_min(1)
                force = force + force_at_temperature / force_norm.clamp_min(1e-8).sqrt()
            target = old_gen_scaled + force

        error = (gen / scale_inputs - target).square().mean(dim=1)
        return error * mask
