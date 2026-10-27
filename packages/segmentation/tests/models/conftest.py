"""Shared fixtures for model tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def dummy_input():
    """Small random input tensor (batch=2, RGB, 32x32) for fast model tests."""
    torch = pytest.importorskip("torch")
    return torch.randn(2, 3, 32, 32)
