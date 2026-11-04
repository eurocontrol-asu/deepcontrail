"""Pretrained segmentation models: save, load (local folder or Hugging Face Hub) and predict.

A pretrained bundle is a folder with two files:

- ``model.safetensors``: the weights of the segmentation network;
- ``config.json`` (``PretrainedConfig``): the architecture, the false-colour recipe
  (band keys of the training data), normalisation, input size, decision threshold
  and whether D4 test-time augmentation is recommended.

It may also carry the instance step tuned for this model, ``instance_pipeline.yaml`` and
``score_model.json``; they are fetched with the bundle and read by ``deepcontrail.instance``
from ``Pretrained.directory``.

Example::

    from deepcontrail.core import false_color_from_spectral, read_band_manifest
    from deepcontrail.segmentation import load_pretrained, predict

    model = load_pretrained("path/or/hub-repo-id")
    composite = false_color_from_spectral(bands, model.config.false_color, manifest)
    probabilities = predict(model, composite, tta=True)
    mask = probabilities > model.config.threshold
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_model, save_model

from deepcontrail.core.false_color import FalseColorConfig  # noqa: TC001
from deepcontrail.core.readers import shift_image
from deepcontrail.segmentation.config import NormalizeConfig
from deepcontrail.segmentation.models import ModelConfig, build_model
from deepcontrail.segmentation.tensors import _norm_tensors
from deepcontrail.segmentation.training.tta import tta_predict

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
INSTANCE_FILES = ("instance_pipeline.yaml", "score_model.json")
"""Optional files of a bundle: the instance step tuned for this model (deepcontrail.instance)."""


class PretrainedConfig(BaseModel):
    """``config.json`` of a pretrained bundle: everything needed to run the weights.

    Parameters
    ----------
    model
        Architecture of the network (its ``encoder_weights`` are ignored on load).
    false_color
        Composite the model was trained on, by the band keys of its training data.
    normalize
        Channel normalisation applied after resizing.
    image_size
        Side length the composite is resized to before the network.
    native_size
        Side length of the training scenes (and their masks); ``predict`` accepts only
        scenes of this size.
    threshold
        Probability above which a pixel is contrail.
    tta
        Whether D4 test-time augmentation is recommended (8x slower).
    metrics
        Evaluation results, e.g. ``{"test_dice_tta_at_0.5": 0.696}``.
    provenance
        Where the weights come from: training run, data export, code version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Literal[1]
    model: ModelConfig
    false_color: FalseColorConfig
    normalize: NormalizeConfig = NormalizeConfig()
    image_size: Annotated[int, Field(gt=0)]
    native_size: Annotated[int, Field(gt=0)]
    threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    tta: bool = True
    metrics: dict[str, float] = {}
    provenance: dict[str, str] = {}


@dataclasses.dataclass
class Pretrained:
    """A segmentation network in eval mode with the config it must be run with.

    ``directory`` is the folder the bundle was loaded from (the local download for a Hub
    repository), where its optional files are.
    """

    model: torch.nn.Module
    config: PretrainedConfig
    directory: Path | None = None

    @property
    def device(self) -> torch.device:
        """Device the network's weights are on."""
        return next(self.model.parameters()).device


def save_pretrained(
    model: torch.nn.Module, config: PretrainedConfig, directory: str | Path
) -> None:
    """Write ``model.safetensors`` and ``config.json`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(directory / WEIGHTS_FILE))
    (directory / CONFIG_FILE).write_text(config.model_dump_json(indent=2) + "\n")


def load_pretrained(
    source: str | Path, *, revision: str | None = None, device: str | torch.device = "cpu"
) -> Pretrained:
    """Load a pretrained bundle from a local folder or a Hugging Face Hub repository.

    Parameters
    ----------
    source
        A folder holding the bundle, or a Hub repository ID (``"org/name"``).
    revision
        Hub revision (tag, branch or commit); ignored for a local folder.
    device
        Device to put the network on.

    Raises
    ------
    FileNotFoundError
        If a bundle file is missing.
    """
    directory = Path(source)
    if not directory.is_dir():
        import huggingface_hub

        directory = Path(
            huggingface_hub.snapshot_download(
                repo_id=str(source),
                revision=revision,
                allow_patterns=[WEIGHTS_FILE, CONFIG_FILE, *INSTANCE_FILES],
            )
        )
    for name in (WEIGHTS_FILE, CONFIG_FILE):
        if not (directory / name).is_file():
            msg = f"{directory / name}: missing from the pretrained bundle"
            raise FileNotFoundError(msg)

    config = PretrainedConfig.model_validate_json((directory / CONFIG_FILE).read_text())
    # The architecture alone: the weights come from model.safetensors.
    model = build_model(config.model.model_copy(update={"encoder_weights": None}))
    load_model(model, str(directory / WEIGHTS_FILE), strict=True)
    return Pretrained(model.to(device).eval(), config, directory)


class _AtInputSize(torch.nn.Module):
    """Run the network, then resize its logits back to the composite's size."""

    def __init__(self, model: torch.nn.Module, size: tuple[int, int]) -> None:
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if tuple(output.shape[-2:]) != self.size:
            output = F.interpolate(output, size=self.size, mode="bilinear", align_corners=False)
        return output


def _prepare(composite: np.ndarray, config: PretrainedConfig, pixel_shift: float) -> torch.Tensor:
    """Composite (H, W, 3) in [0, 1] -> normalised (3, image_size, image_size) tensor.

    The same steps as ``SegmentationDataset``: pixel shift, albumentations resize,
    channel normalisation.
    """
    image = composite.astype(np.float32)
    if pixel_shift:
        image = shift_image(image, pixel_shift)
    if image.shape[0] != config.image_size or image.shape[1] != config.image_size:
        from deepcontrail.segmentation.transforms import build_image_resize

        image = build_image_resize(config.image_size)(image=image)["image"]
    mean, std = _norm_tensors(config.normalize.mean, config.normalize.std)
    return (torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1) - mean) / std


@torch.no_grad()
def predict(
    pretrained: Pretrained,
    composite: np.ndarray,
    *,
    tta: bool = False,
    pixel_shift: float = 0.0,
    batch_size: int = 8,
) -> np.ndarray:
    """Return contrail probabilities for one composite or a batch of composites.

    Parameters
    ----------
    pretrained
        A model from ``load_pretrained``.
    composite
        ``(H, W, 3)`` or ``(N, H, W, 3)`` float array in ``[0, 1]``, built from the
        bands with ``pretrained.config.false_color``. ``H`` and ``W`` must equal
        ``pretrained.config.native_size``, the size of the scenes the model was trained on.
    tta
        Average the 8 D4 transforms (see ``pretrained.config.tta``).
    pixel_shift
        Shift (pixels, right and down) aligning the input with the labels the model
        was trained on; 0 when the input is already aligned.
    batch_size
        Composites per forward pass.

    Returns
    -------
    np.ndarray
        Float32 probabilities, ``(H, W)`` or ``(N, H, W)``, at the input's resolution.

    Raises
    ------
    ValueError
        If the composite is not 3-channel, or its scenes are not ``native_size`` square:
        resizing another size to the network's input would change the pixel scale the model
        was trained on (larger scenes are not tiled yet).
    """
    single = composite.ndim == 3
    batch = composite[None] if single else composite
    if batch.ndim != 4 or batch.shape[-1] != 3:
        msg = f"expected a (H, W, 3) or (N, H, W, 3) composite, got shape {composite.shape}"
        raise ValueError(msg)
    size = pretrained.config.native_size
    if batch.shape[1:3] != (size, size):
        msg = (
            f"the model was trained on {size} x {size} scenes, got {batch.shape[1]} x "
            f"{batch.shape[2]}: other sizes would change the pixel scale (tiling is not supported)"
        )
        raise ValueError(msg)

    network = _AtInputSize(pretrained.model, (batch.shape[1], batch.shape[2])).eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(batch), batch_size):
        images = torch.stack(
            [_prepare(c, pretrained.config, pixel_shift) for c in batch[start : start + batch_size]]
        ).to(pretrained.device)
        probs = tta_predict(network, images) if tta else torch.sigmoid(network(images)).squeeze(1)
        outputs.append(probs.float().cpu().numpy())
    result: np.ndarray[Any, Any] = np.concatenate(outputs).astype(np.float32)
    return result[0] if single else result
