"""Shared fixtures for loss tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def logits_and_targets():
    """Small logit/target pair for loss tests (batch=2, 1-class, 8x8)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
    return logits, targets
