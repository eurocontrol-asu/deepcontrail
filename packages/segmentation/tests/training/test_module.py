"""Tests for SegmentationModule forward-pass resizing."""

from __future__ import annotations

import pytest
import torch

from deepcontrail.segmentation.training import SegmentationModule


class _DummySeg(torch.nn.Module):
    """A 1x1 convolution: output spatial size mirrors the input size."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _module(native_size: int = 256) -> SegmentationModule:
    model = _DummySeg()
    # A loss_fn is required by the module even though forward() ignores it.
    module = SegmentationModule(
        model=model,
        loss_fn=torch.nn.Identity(),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=native_size,
    )
    module.eval()
    return module


def test_forward_downscaled_output_is_upsampled_to_native():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 384, 384))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_upscaled_output_is_downsampled_to_native():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 512, 512))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_native_input_passes_through_unchanged():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 256, 256))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_custom_native_size():
    module = _module(native_size=128)

    out = module(torch.randn(1, 3, 384, 384))

    assert out.shape == (1, 1, 128, 128), f"got {out.shape}"


def test_forward_output_shape_matches_mask_resolution():
    """The resized output is what loss/metrics (at mask resolution) consume."""
    module = _module(native_size=256)

    images = torch.randn(4, 3, 512, 512)  # image_size = 512
    masks = torch.randint(0, 2, (4, 256, 256))  # native mask resolution

    out = module(images)

    # Output (B,1,256,256) pairs with mask (B,256,256) for loss/metric calls.
    assert out.shape == (4, 1, 256, 256)
    assert masks.shape == (4, 256, 256)


def test_module_requires_the_native_size() -> None:
    import torch

    from deepcontrail.segmentation.training.module import SegmentationModule

    with pytest.raises(TypeError, match="native_size"):
        SegmentationModule(  # type: ignore[call-arg]
            model=torch.nn.Identity(),
            loss_fn=torch.nn.Identity(),
            lr=1e-3,
            weight_decay=0.0,
            scheduler="none",
            epochs=1,
        )


def test_training_step_logs_the_epoch_mean_and_the_per_step_loss(monkeypatch) -> None:
    from deepcontrail.segmentation.losses import LossConfig, build_loss

    module = SegmentationModule(
        model=_DummySeg(),
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=8,
    )
    logged: dict[str, dict] = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.setdefault(name, kw))
    batch = (torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))

    module.training_step(batch, 0)

    assert logged["train_loss"]["on_epoch"] is True, "train_loss stays the per-epoch mean"
    assert logged["train_loss"]["on_step"] is False
    assert logged["train_loss_step"]["on_step"] is True, "the per-step loss gives the fine curve"
    assert logged["train_loss_step"]["on_epoch"] is False


def _bce_module() -> SegmentationModule:
    from deepcontrail.segmentation.losses import LossConfig, build_loss

    return SegmentationModule(
        model=_DummySeg(),
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=4,
    )


def _train_epoch(module: SegmentationModule, batches, monkeypatch) -> dict[str, float]:
    """Run training steps whose logits are given, then end the epoch; return logged values."""
    logged: dict[str, float] = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))
    for logits, masks in batches:
        monkeypatch.setattr(module, "forward", lambda images, logits=logits: logits)
        module.training_step((torch.zeros(1, 3, 4, 4), masks), 0)
    module.on_train_epoch_end()
    return logged


def _logits(pattern: list[list[int]]) -> torch.Tensor:
    """Logits whose sigmoid is above 0.5 exactly where pattern is 1."""
    return (torch.tensor(pattern, dtype=torch.float32) * 20 - 10).view(1, 1, 4, 4)


_HALF = [[1, 1, 0, 0]] * 4


def test_train_epoch_logs_global_dice_and_iou(monkeypatch) -> None:
    module = _bce_module()
    masks = torch.tensor(_HALF, dtype=torch.long).view(1, 4, 4)

    logged = _train_epoch(module, [(_logits(_HALF), masks)], monkeypatch)

    assert logged["train_dice"] == pytest.approx(1.0), "perfect predictions give Dice 1"
    assert logged["train_iou"] == pytest.approx(1.0)


def test_train_metrics_are_global_over_the_epoch(monkeypatch) -> None:
    module = _bce_module()
    target = torch.tensor(_HALF, dtype=torch.long).view(1, 4, 4)
    all_on = [[1, 1, 1, 1]] * 4

    # Batch 1 exact (8 TP); batch 2 predicts all 16 pixels for 8 targets (8 TP, 8 FP).
    logged = _train_epoch(
        module, [(_logits(_HALF), target), (_logits(all_on), target)], monkeypatch
    )

    tp, pred, true = 16, 8 + 16, 8 + 8
    assert logged["train_dice"] == pytest.approx(2 * tp / (pred + true), rel=1e-5)
    assert logged["train_iou"] == pytest.approx(tp / (pred + true - tp), rel=1e-5)


def test_train_metrics_count_soft_targets_above_half_as_contrail(monkeypatch) -> None:
    module = _bce_module()
    soft = torch.tensor([[0.75, 0.75, 0.25, 0.25]] * 4).view(1, 4, 4)

    logged = _train_epoch(module, [(_logits(_HALF), soft)], monkeypatch)

    assert logged["train_dice"] == pytest.approx(1.0), (
        "a pixel is contrail when more than half the annotators marked it"
    )


def test_train_metrics_reset_between_epochs(monkeypatch) -> None:
    module = _bce_module()
    target = torch.tensor(_HALF, dtype=torch.long).view(1, 4, 4)
    _train_epoch(module, [(_logits([[0, 0, 1, 1]] * 4), target)], monkeypatch)

    logged = _train_epoch(module, [(_logits(_HALF), target)], monkeypatch)

    assert logged["train_dice"] == pytest.approx(1.0), "each epoch starts from zero counts"


class _TwoPartModel(torch.nn.Module):
    """A tiny SMP-style model with separate ``encoder`` and ``decoder`` submodules."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.decoder = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _two_part_module(**overrides: object) -> SegmentationModule:
    """Build a ``SegmentationModule`` wrapping a model that exposes ``.encoder``."""
    defaults: dict[str, object] = {
        "model": _TwoPartModel(),
        "loss_fn": torch.nn.Identity(),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "epochs": 10,
        "native_size": 8,
    }
    defaults.update(overrides)
    return SegmentationModule(**defaults)  # type: ignore[arg-type]


def _set_epoch(module: SegmentationModule, epoch: int) -> None:
    """Point the module's read-only ``current_epoch`` at a fake trainer (tests only)."""
    from types import SimpleNamespace

    module._trainer = SimpleNamespace(current_epoch=epoch)  # type: ignore[attr-defined]


def test_configure_optimizers_default_has_one_param_group() -> None:
    module = _two_part_module()

    groups = module.configure_optimizers()["optimizer"].param_groups

    assert len(groups) == 1, (
        "encoder_lr_scale defaults to 1.0: a single param group, i.e. today's behaviour"
    )


def test_configure_optimizers_encoder_lr_scale_gives_two_groups_with_scaled_lr() -> None:
    lr, scale = 1e-3, 0.1
    module = _two_part_module(lr=lr, encoder_lr_scale=scale)

    groups = module.configure_optimizers()["optimizer"].param_groups

    assert len(groups) == 2, "a scaled encoder splits the parameters into two groups"
    id_sets = [{id(p) for p in group["params"]} for group in groups]
    assert id_sets[0].isdisjoint(id_sets[1]), "every parameter is in exactly one group"
    assert id_sets[0] | id_sets[1] == {id(p) for p in module.model.parameters()}, (
        "the two groups together cover every model parameter, none dropped"
    )
    base_lrs = sorted(group["initial_lr"] for group in groups)
    assert base_lrs == [lr * scale, lr], (
        f"the encoder group scales its base lr (lr*scale) and the rest keeps lr, got {base_lrs}"
    )
    encoder_ids = {id(p) for p in module._encoder_params()}
    encoder_group = next(group for group in groups if group["initial_lr"] == lr * scale)
    assert {id(p) for p in encoder_group["params"]} == encoder_ids, (
        "the scaled group holds the encoder parameters"
    )


def test_freeze_encoder_epochs_stops_encoder_updates() -> None:
    module = _two_part_module(freeze_encoder_epochs=2)
    encoder = module.model.encoder

    _set_epoch(module, 0)  # inside the frozen prefix
    module.on_train_epoch_start()

    assert all(not p.requires_grad for p in encoder.parameters()), (
        "a frozen encoder receives no gradients"
    )
    assert not encoder.training, (
        "a frozen encoder is in eval mode, so its BatchNorm/LayerNorm stats do not update"
    )

    _set_epoch(module, 2)  # reaching freeze_encoder_epochs thaws it
    module.on_train_epoch_start()

    assert all(p.requires_grad for p in encoder.parameters()), (
        "once current_epoch reaches freeze_encoder_epochs the encoder trains again"
    )


def test_model_without_encoder_and_scale_raises() -> None:
    module = SegmentationModule(
        model=torch.nn.Conv2d(3, 1, 1),  # no ``.encoder`` attribute
        loss_fn=torch.nn.Identity(),
        lr=1e-3,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=10,
        native_size=8,
        encoder_lr_scale=0.1,
    )

    with pytest.raises(ValueError, match="encoder"):
        module.configure_optimizers()


def test_training_step_creates_no_float64_tensor(monkeypatch) -> None:
    from torch.overrides import TorchFunctionMode

    class _RecordDtypes(TorchFunctionMode):
        def __init__(self) -> None:
            super().__init__()
            self.dtypes: set[torch.dtype] = set()

        def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor):
                self.dtypes.add(out.dtype)
            return out

    module = _bce_module()
    masks = torch.tensor(_HALF, dtype=torch.long).view(1, 4, 4)
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "forward", lambda images: _logits(_HALF))
    recorder = _RecordDtypes()

    with recorder:
        module.training_step((torch.zeros(1, 3, 4, 4), masks), 0)

    assert torch.float64 not in recorder.dtypes, (
        "Apple MPS devices have no float64: a training step must not create float64 tensors"
    )


# --- trainable (partial freezing) ------------------------------------------------------


class _NamedSubmoduleModel(torch.nn.Module):
    """A tiny SMP-style model whose parameter names mirror the real one.

    ``encoder.stem`` (a conv) and ``encoder.body`` (a BatchNorm then a conv) feed ``decoder``
    and ``segmentation_head`` (convs); ``forward`` chains them so shapes work at any resolution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleDict(
            {
                "stem": torch.nn.Conv2d(3, 4, 3, padding=1),
                "body": torch.nn.Sequential(
                    torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 4, 3, padding=1)
                ),
            }
        )
        self.decoder = torch.nn.Conv2d(4, 1, 1)
        self.segmentation_head = torch.nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder["stem"](x)
        x = self.encoder["body"](x)
        x = self.decoder(x)
        return self.segmentation_head(x)


def _named_module(**overrides: object) -> SegmentationModule:
    """Build a ``SegmentationModule`` around ``_NamedSubmoduleModel`` (encoder/decoder/head)."""
    defaults: dict[str, object] = {
        "model": _NamedSubmoduleModel(),
        "loss_fn": torch.nn.Identity(),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "epochs": 5,
        "native_size": 8,
    }
    defaults.update(overrides)
    return SegmentationModule(**defaults)  # type: ignore[arg-type]


def test_trainable_none_keeps_every_parameter_trainable_and_one_group() -> None:
    module = _named_module()

    assert all(p.requires_grad for p in module.model.parameters()), (
        "trainable None must leave every parameter trainable, i.e. today's behaviour"
    )

    groups = module.configure_optimizers()["optimizer"].param_groups
    assert len(groups) == 1, "trainable None keeps the single optimizer param group"
    covered = {id(p) for group in groups for p in group["params"]}
    assert covered == {id(p) for p in module.model.parameters()}, (
        "every parameter stays in the one group when nothing is frozen"
    )


def test_trainable_prefixes_only_those_get_gradients() -> None:
    module = _named_module(trainable=["decoder"])
    module.train()
    images = torch.randn(2, 3, 8, 8)

    module.model(images).sum().backward()

    assert module.model.decoder.weight.grad is not None, "a trainable parameter receives gradients"
    stem = module.model.encoder["stem"]
    assert not stem.weight.requires_grad, "a frozen parameter has requires_grad False"
    assert stem.weight.grad is None, "a frozen parameter receives no gradients"


def test_trainable_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError, match="decoder_typo"):
        _named_module(trainable=["decoder_typo"])  # a prefix must match a real parameter


def test_trainable_prefix_matches_whole_name_components() -> None:
    with pytest.raises(ValueError, match="decod"):
        _named_module(trainable=["decod"])  # "decod" must NOT match "decoder.weight"


def test_trainable_frozen_modules_are_in_eval_mode_after_epoch_start() -> None:
    module = _named_module(trainable=["decoder", "segmentation_head"])
    batch_norm = module.model.encoder["body"][0]  # a BatchNorm under the frozen encoder
    module.train()  # Lightning sets the whole module to train mode at every epoch start
    with torch.no_grad():  # non-zero running stats: a train-mode pass would move them
        batch_norm.running_mean.fill_(1.0)
        batch_norm.running_var.fill_(2.0)

    module.on_train_epoch_start()

    assert not batch_norm.training, (
        "a module under a frozen prefix is put in eval mode, so its stats do not update"
    )
    assert module.model.decoder.training, "a trainable module stays in train mode"

    module(torch.randn(2, 3, 8, 8))  # a forward pass in train mode

    assert torch.equal(batch_norm.running_mean, torch.full_like(batch_norm.running_mean, 1.0)), (
        "an eval-mode BatchNorm keeps its running statistics through a forward pass"
    )


def test_configure_optimizers_trainable_holds_only_trainable_params() -> None:
    module = _named_module(trainable=["decoder"])

    params = module.configure_optimizers()["optimizer"].param_groups[0]["params"]
    held = {id(p) for p in params}

    assert held == {id(p) for p in module.model.decoder.parameters()}, (
        "only the trainable (decoder) parameters go into the optimizer"
    )
    assert all(p.requires_grad for p in params), "every held parameter is trainable"
    assert held.isdisjoint({id(p) for p in module.model.encoder.parameters()}), (
        "no frozen (encoder) parameter is in the optimizer"
    )


def test_configure_optimizers_trainable_and_encoder_lr_scale_drops_empty_groups() -> None:
    module = _two_part_module(trainable=["decoder"], encoder_lr_scale=0.1)

    groups = module.configure_optimizers()["optimizer"].param_groups

    assert len(groups) == 1, "an all-frozen encoder group is dropped, leaving the trainable one"
    held = {id(p) for p in groups[0]["params"]}
    assert held == {id(p) for p in module.model.decoder.parameters()}, (
        "the surviving group holds only the trainable decoder parameters"
    )


def test_trainable_reports_the_frozen_count(capsys: pytest.CaptureFixture[str]) -> None:
    _named_module(trainable=["decoder"])

    printed = capsys.readouterr().out

    assert "Trainable weights: 2 of 10 (20.0%), frozen 8" in printed, (
        "every run reports how many weights it trains and how many it freezes"
    )
