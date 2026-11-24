"""Validation with a fed memory: the module feeds each item its own prediction for the previous frame."""

from __future__ import annotations

import torch


class _Recorder(torch.nn.Module):
    """Return logits that vary across the width, and keep every input it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        ramp = torch.linspace(-2.0, 2.0, x.shape[-1])[None, None, None, :]
        return x[:, :1] * 0 + ramp * self.scale


def _module() -> tuple[object, _Recorder]:
    from deepcontrail.segmentation.losses import LossConfig, build_loss
    from deepcontrail.segmentation.training import SegmentationModule

    recorder = _Recorder()
    module = SegmentationModule(
        model=recorder,
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-3,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=8,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]  # no Trainer here
    return module, recorder


def _batch(index: int, prev: int) -> tuple[torch.Tensor, ...]:
    images = torch.zeros(1, 4, 8, 8)  # 3 image channels + the (empty) memory channel
    masks = torch.zeros(1, 8, 8, dtype=torch.long)
    return images, masks, torch.tensor([index]), torch.tensor([prev])


def test_validation_feeds_each_item_the_prediction_for_its_previous_frame():
    module, recorder = _module()
    expected = torch.sigmoid(torch.linspace(-2.0, 2.0, 8))[None, :].expand(8, 8)

    module.on_validation_epoch_start()
    module.validation_step(_batch(0, -1), 0)
    module.validation_step(_batch(1, 0), 1)

    assert torch.equal(recorder.inputs[0][0, -1], torch.zeros(8, 8)), "a sequence start: no memory"
    torch.testing.assert_close(
        recorder.inputs[1][0, -1], expected, msg="the memory is the model's prediction for item 0"
    )
    assert torch.equal(recorder.inputs[1][0, :3], torch.zeros(3, 8, 8)), "image channels untouched"


def test_validation_memory_restarts_empty_at_each_epoch():
    module, recorder = _module()

    module.on_validation_epoch_start()
    module.validation_step(_batch(0, -1), 0)
    module.on_validation_epoch_start()
    module.validation_step(_batch(1, 0), 0)

    assert torch.equal(recorder.inputs[1][0, -1], torch.zeros(8, 8)), (
        "a new epoch does not reuse the previous epoch's predictions"
    )


def test_validation_without_indices_is_unchanged():
    module, recorder = _module()
    images = torch.rand(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))

    module.on_validation_epoch_start()
    module.validation_step((images, torch.zeros(1, 8, 8, dtype=torch.long)), 0)

    assert torch.equal(recorder.inputs[0], images), "a plain (image, mask) batch goes in as is"
