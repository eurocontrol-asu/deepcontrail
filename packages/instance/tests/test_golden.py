"""Golden regression test: exact instance masks, scores and score-model features on fixed scenes.

Any change to the instance results must be deliberate: regenerate the fixture with
``uv run --no-sync python packages/instance/tests/test_golden.py`` in the commit that changes them,
and say why in its message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepcontrail.instance import (
    InstancePipelineConfig,
    MannsteinConfig,
    detect_instances,
    detect_instances_from_probabilities,
)
from deepcontrail.instance.scoring import scene_features

FIXTURE = Path(__file__).with_name("golden_instances.npz")
_SIZE = 96


def _scene(seed: int) -> np.ndarray:
    """Build a probability map with crossing, near-axis and diagonal contrails, soft edges."""
    rng = np.random.default_rng(seed)
    probs = np.zeros((_SIZE, _SIZE), dtype=np.float32)
    t = np.linspace(-40.0, 40.0, 600)
    for angle in (30.0, 120.0, 4.0, 87.0, 55.0):
        theta = np.radians(angle + rng.uniform(-2.0, 2.0))
        r0, c0 = rng.uniform(30.0, 66.0, size=2)
        for offset, value in ((0.0, 0.9), (-1.0, 0.45), (1.0, 0.45)):
            rows = np.round(r0 - t * np.sin(theta) + offset * np.cos(theta)).astype(int)
            cols = np.round(c0 + t * np.cos(theta) + offset * np.sin(theta)).astype(int)
            keep = (rows >= 0) & (rows < _SIZE) & (cols >= 0) & (cols < _SIZE)
            probs[rows[keep], cols[keep]] = np.maximum(probs[rows[keep], cols[keep]], value)
    return probs


def _outputs() -> dict[str, np.ndarray]:
    """Run the three reference settings on both scenes; flatten masks to sorted pixel ids."""
    out: dict[str, np.ndarray] = {}
    settings = {
        "defaults": lambda p: detect_instances_from_probabilities(p, InstancePipelineConfig()),
        "consensus": lambda p: detect_instances_from_probabilities(
            p, InstancePipelineConfig.consensus()
        ),
        "plain": lambda p: detect_instances((p > 0.5).astype(np.uint8), MannsteinConfig()),
    }
    for seed in (0, 1):
        probs = _scene(seed)
        for name, run in settings.items():
            masks, scores = run(probs)
            key = f"{name}_{seed}"
            ids = [np.flatnonzero(m) for m in masks]
            out[f"{key}_lengths"] = np.array([len(i) for i in ids], dtype=np.int64)
            out[f"{key}_pixels"] = (
                np.concatenate(ids).astype(np.int64) if ids else np.zeros(0, dtype=np.int64)
            )
            out[f"{key}_scores"] = np.asarray(scores, dtype=np.float64)
            out[f"{key}_features"] = scene_features(masks, probs, scores)
    return out


def test_instance_results_match_the_golden_fixture():
    expected = np.load(FIXTURE)

    actual = _outputs()

    assert set(actual) == set(expected.files)
    for key, value in actual.items():
        if key.endswith(("_scores", "_features")):
            np.testing.assert_allclose(value, expected[key], rtol=0, atol=1e-12, err_msg=key)
        else:
            np.testing.assert_array_equal(value, expected[key], err_msg=key)


if __name__ == "__main__":
    np.savez(FIXTURE, **_outputs())
    print(f"wrote {FIXTURE}")
