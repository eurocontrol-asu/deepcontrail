"""Tests for warmup + cosine LR scheduling in the training module."""

from __future__ import annotations

from torch import nn

from deepcontrail.segmentation.training import SegmentationModule, TrainingConfig


def _dummy_module(
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    warmup_epochs: int = 0,
    scheduler_interval: str = "epoch",
    steps_per_epoch: int = 100,
) -> SegmentationModule:
    """Build a minimal SegmentationModule for scheduler testing."""
    model = nn.Conv2d(3, 1, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    return SegmentationModule(
        model=model,
        loss_fn=loss_fn,
        lr=lr,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=epochs,
        native_size=16,
        warmup_epochs=warmup_epochs,
        scheduler_interval=scheduler_interval,
        steps_per_epoch=steps_per_epoch,
    )


def _collect_lrs(module: SegmentationModule, num_steps: int) -> list[float]:
    """Step the scheduler num_steps times and collect LRs."""
    opt_cfg = module.configure_optimizers()
    optimizer = opt_cfg["optimizer"]
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    lrs = []
    for _ in range(num_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


# --- TrainingConfig field tests ---


def test_training_config_default_warmup_epochs_is_zero() -> None:
    cfg = TrainingConfig()

    assert cfg.warmup_epochs == 0, "warmup_epochs should default to 0"


def test_training_config_default_scheduler_interval_is_epoch() -> None:
    cfg = TrainingConfig()

    assert cfg.scheduler_interval == "epoch", "scheduler_interval should default to 'epoch'"


def test_training_config_from_dict_with_warmup_fields() -> None:
    cfg = TrainingConfig.from_dict({"warmup_epochs": 5, "scheduler_interval": "step"})

    assert cfg.warmup_epochs == 5
    assert cfg.scheduler_interval == "step"


# --- Backward compatibility ---


def test_no_warmup_produces_pure_cosine() -> None:
    """warmup_epochs=0 must produce the same LR curve as plain CosineAnnealingLR."""
    module = _dummy_module(epochs=10, lr=1e-3, warmup_epochs=0)

    opt_cfg = module.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert type(scheduler).__name__ == "CosineAnnealingLR", (
        "With warmup_epochs=0, scheduler should be plain CosineAnnealingLR"
    )


def test_no_warmup_interval_is_epoch() -> None:
    module = _dummy_module(warmup_epochs=0, scheduler_interval="epoch")

    opt_cfg = module.configure_optimizers()
    interval = opt_cfg["lr_scheduler"]["interval"]

    assert interval == "epoch"


# --- Warmup behavior ---


def test_warmup_produces_sequential_scheduler() -> None:
    module = _dummy_module(warmup_epochs=2, scheduler_interval="step")

    opt_cfg = module.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert type(scheduler).__name__ == "SequentialLR", (
        "With warmup_epochs>0, scheduler should be SequentialLR"
    )


def test_warmup_interval_is_step() -> None:
    module = _dummy_module(warmup_epochs=2, scheduler_interval="step")

    opt_cfg = module.configure_optimizers()
    interval = opt_cfg["lr_scheduler"]["interval"]

    assert interval == "step"


def test_lr_starts_near_zero_with_warmup() -> None:
    lr = 1e-3
    module = _dummy_module(
        epochs=10,
        lr=lr,
        warmup_epochs=2,
        scheduler_interval="step",
        steps_per_epoch=50,
    )

    lrs = _collect_lrs(module, num_steps=1)

    assert lrs[0] < lr * 0.05, f"LR should start near zero with warmup, got {lrs[0]}"


def test_lr_reaches_peak_at_end_of_warmup() -> None:
    lr = 1e-3
    warmup_epochs = 2
    steps_per_epoch = 50
    warmup_steps = warmup_epochs * steps_per_epoch
    module = _dummy_module(
        epochs=10,
        lr=lr,
        warmup_epochs=warmup_epochs,
        scheduler_interval="step",
        steps_per_epoch=steps_per_epoch,
    )

    lrs = _collect_lrs(module, num_steps=warmup_steps + 1)

    assert abs(lrs[warmup_steps] - lr) < lr * 0.05, (
        f"LR should reach peak ({lr}) at step {warmup_steps}, got {lrs[warmup_steps]}"
    )


def test_lr_monotonically_increases_during_warmup() -> None:
    lr = 1e-3
    warmup_epochs = 2
    steps_per_epoch = 50
    warmup_steps = warmup_epochs * steps_per_epoch
    module = _dummy_module(
        epochs=10,
        lr=lr,
        warmup_epochs=warmup_epochs,
        scheduler_interval="step",
        steps_per_epoch=steps_per_epoch,
    )

    lrs = _collect_lrs(module, num_steps=warmup_steps)

    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i - 1], (
            f"LR should increase during warmup, but step {i} ({lrs[i]}) "
            f"< step {i - 1} ({lrs[i - 1]})"
        )


def test_lr_decays_after_warmup() -> None:
    lr = 1e-3
    warmup_epochs = 2
    steps_per_epoch = 50
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = 10 * steps_per_epoch
    module = _dummy_module(
        epochs=10,
        lr=lr,
        warmup_epochs=warmup_epochs,
        scheduler_interval="step",
        steps_per_epoch=steps_per_epoch,
    )

    lrs = _collect_lrs(module, num_steps=total_steps)

    peak_lr = lrs[warmup_steps]
    final_lr = lrs[-1]
    assert final_lr < peak_lr * 0.1, (
        f"LR should decay to near zero after cosine phase, got final={final_lr}, peak={peak_lr}"
    )


def test_warmup_epoch_interval_uses_epoch_units() -> None:
    """When interval='epoch', warmup iters should be in epochs, not steps."""
    module = _dummy_module(
        epochs=10,
        lr=1e-3,
        warmup_epochs=3,
        scheduler_interval="epoch",
        steps_per_epoch=100,
    )

    opt_cfg = module.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert type(scheduler).__name__ == "SequentialLR"
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"

    lrs = _collect_lrs(module, num_steps=10)
    assert abs(lrs[3] - 1e-3) < 1e-3 * 0.05, (
        f"At epoch 3 (end of warmup), LR should be near peak, got {lrs[3]}"
    )
