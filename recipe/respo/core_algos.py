# Copyright 2024 VESPO Authors
# Copyright 2026 ReSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ReSPO's two-branch sequence-level policy loss.

Importing this module registers ``loss_mode=respo`` with verl. The constants
below are the hyperparameters used for every experiment reported in the paper.
"""

from typing import Any, Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

ALPHA_POS = 2.0
BETA_POS = 0.5
LAMBDA_POS = 2.0
ALPHA_NEG = 1.0
BETA_NEG = 0.5
LAMBDA_NEG = 2.0

__all__ = ["compute_respo_phi", "compute_policy_loss_respo"]


def compute_respo_phi(log_w: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Compute the detached ReSPO coefficient for each response.

    ``log_w`` and ``advantages`` are one-dimensional tensors with one value per
    response. The implementation is the paper's alpha-positive=2 and
    alpha-negative=1 specialization, evaluated after clamping log W to
    ``[-20, 20]``.
    """
    log_w = log_w.detach().clamp(-20.0, 20.0)
    w = log_w.exp()

    # Positive branch: phi_0(W) = (1 + W) / 2.
    phi0_pos = (1.0 - BETA_POS) + BETA_POS * w
    phi_pos = phi0_pos * torch.exp(LAMBDA_POS * (1.0 - phi0_pos))

    # Negative branch: alpha -> 1 gives phi_0(W) = W^beta.
    log_phi0_neg = BETA_NEG * log_w
    phi0_neg = log_phi0_neg.exp()
    log_phi_neg = log_phi0_neg + LAMBDA_NEG * (1.0 - phi0_neg)
    phi_neg = log_phi_neg.exp()

    phi = torch.where(advantages >= 0, phi_pos, phi_neg)
    return torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0).detach()


@register_policy_loss("respo")
def compute_policy_loss_respo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the ReSPO policy-gradient loss.

    The response-level coefficient is detached and scales every valid token in
    that response. ``rollout_is_weights`` is incorporated in log space when
    verl supplies rollout-policy correction weights.
    """
    if config is None:
        raise ValueError("ReSPO requires verl's ActorConfig")

    mask = response_mask.to(log_prob.dtype)
    token_log_ratio = log_prob - old_log_prob
    raw_sequence_log_ratio = torch.sum(token_log_ratio * mask, dim=-1)
    sequence_log_ratio = raw_sequence_log_ratio

    if rollout_is_weights is not None:
        rollout_log_ratio = torch.log(rollout_is_weights.clamp_min(1e-8))
        sequence_log_ratio = sequence_log_ratio + torch.sum(rollout_log_ratio * mask, dim=-1)

    sequence_log_ratio = sequence_log_ratio.clamp(-20.0, 20.0)
    sequence_advantage = advantages[:, 0] if advantages.ndim > 1 else advantages
    phi = compute_respo_phi(sequence_log_ratio, sequence_advantage)

    loss_matrix = -phi.unsqueeze(-1) * advantages * log_prob
    policy_loss = agg_loss(
        loss_mat=loss_matrix,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    valid_lengths = mask.sum(dim=-1).clamp_min(1.0)
    mean_token_log_ratio = raw_sequence_log_ratio / valid_lengths
    metrics = {
        "actor/approx_kl": (-mean_token_log_ratio).mean().detach().item(),
        "respo/phi_mean": phi.mean().item(),
        "respo/phi_min": phi.min().item(),
        "respo/phi_max": phi.max().item(),
    }
    return policy_loss, metrics
