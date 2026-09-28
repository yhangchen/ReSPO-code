# Copyright 2026 ReSPO Authors
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace

import torch

from recipe.respo.code.core_algos import compute_policy_loss_respo, compute_respo_phi


def test_phi_is_normalized_at_one():
    advantages = torch.tensor([1.0, -1.0])
    phi = compute_respo_phi(torch.zeros(2), advantages)
    torch.testing.assert_close(phi, torch.ones(2))


def test_two_branches_have_the_paper_tail_behavior():
    log_w = torch.tensor([-20.0, 0.0, 20.0])
    positive_phi = compute_respo_phi(log_w, torch.ones(3))
    negative_phi = compute_respo_phi(log_w, -torch.ones(3))

    assert positive_phi[0] > 1.0
    assert positive_phi[2] < 1e-6
    assert negative_phi[0] < negative_phi[1]
    assert negative_phi[2] < negative_phi[1]


def test_loss_uses_one_detached_coefficient_per_sequence():
    old_log_prob = torch.zeros((2, 2))
    log_prob = torch.tensor([[0.1, 0.2], [-0.2, -0.1]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mask = torch.ones_like(advantages, dtype=torch.bool)
    config = SimpleNamespace(global_batch_info={})

    loss, metrics = compute_policy_loss_respo(
        old_log_prob,
        log_prob,
        advantages,
        mask,
        config=config,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert log_prob.grad is not None
    torch.testing.assert_close(log_prob.grad[0, 0], log_prob.grad[0, 1])
    torch.testing.assert_close(log_prob.grad[1, 0], log_prob.grad[1, 1])
    assert set(metrics) == {"actor/approx_kl", "respo/phi_mean", "respo/phi_min", "respo/phi_max"}
