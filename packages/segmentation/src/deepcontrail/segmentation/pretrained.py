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
    probabilities = predict(model, composite)  # with TTA when the model's config says so
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
from deepcontrail.segmentation.dataset import tile_origins
from deepcontrail.segmentation.models import ModelConfig, build_model
from deepcontrail.segmentation.tensors import _normalized
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
    context_frames
        ``k`` frames before and after the centre composite were stacked with it during training,
        so ``predict`` accepts ``3*(2k+1)`` channels and the composite's frames must be built the
        same way the training data stacked them (see
        :func:`deepcontrail.core.contract.context_ids`). Default 0 is the plain 3-channel composite.
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
    context_frames: Annotated[int, Field(ge=0)] = 0
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


def _blend_weights(tile: int) -> np.ndarray:
    """Tent weights: tile centres count most, edges (less context) least, never zero.

    A ``(tile, tile)`` float32 array whose value at a pixel is the product of a 1-D ramp
    along each axis. Used to blend overlapping native-size tiles: a pixel shared by several
    tiles is a weighted average whose weights sum to one, so a pixel covered by a single tile
    is returned unchanged.
    """
    ramp = np.minimum(np.arange(tile) + 0.5, tile - np.arange(tile) - 0.5) / (tile / 2)
    return np.outer(ramp, ramp).astype(np.float32)


def _prepare(composite: np.ndarray, config: PretrainedConfig, pixel_shift: float) -> torch.Tensor:
    """Composite (H, W, C) in [0, 1] -> normalised (C, image_size, image_size) tensor.

    The same steps as ``SegmentationDataset``: pixel shift, albumentations resize, channel
    normalisation. ``C`` is ``3*(2k+1)`` when ``config.context_frames == k`` (a temporal stack),
    and the mean/std are tiled across those channel groups by ``_normalized`` — the same helper the
    dataset uses, so any channel count works.
    """
    image = composite.astype(np.float32)
    if pixel_shift:
        image = shift_image(image, pixel_shift)
    if image.shape[0] != config.image_size or image.shape[1] != config.image_size:
        from deepcontrail.segmentation.transforms import build_image_resize

        image = build_image_resize(config.image_size)(image=image)["image"]
    image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return _normalized(image_t, config.normalize)


def _prepare_item(
    composite: np.ndarray, memory: np.ndarray | None, config: PretrainedConfig, pixel_shift: float
) -> torch.Tensor:
    """``_prepare`` the composite, then append the memory map (if any) as the last channel.

    The memory is a probability map on the label grid: neither shifted nor normalised, only resized
    (bilinear) like the image when ``image_size`` differs from its size.
    """
    image_t = _prepare(composite, config, pixel_shift)
    if memory is None:
        return image_t
    memory_t = torch.from_numpy(np.ascontiguousarray(memory, dtype=np.float32))[None]
    if tuple(memory_t.shape[-2:]) != tuple(image_t.shape[-2:]):
        memory_t = F.interpolate(
            memory_t[None], size=tuple(image_t.shape[-2:]), mode="bilinear", align_corners=False
        )[0]
    return torch.cat([image_t, memory_t], dim=0)


@torch.no_grad()
def predict(
    pretrained: Pretrained,
    composite: np.ndarray,
    *,
    memory: np.ndarray | None = None,
    tta: bool | None = None,
    pixel_shift: float = 0.0,
    batch_size: int = 8,
) -> np.ndarray:
    """Return contrail probabilities for one composite or a batch of composites.

    Parameters
    ----------
    pretrained
        A model from ``load_pretrained``.
    composite
        ``(H, W, C)`` or ``(N, H, W, C)`` float array in ``[0, 1]``, built from the
        bands with ``pretrained.config.false_color``, where ``C`` is ``3*(2k+1)`` and matches
        ``pretrained.config.context_frames == k`` (``k`` context frames stacked each side, as the
        training data did). ``H`` and ``W`` must be at least ``pretrained.config.native_size``, the
        size of the scenes the model was trained on. A scene exactly that size is predicted as-is;
        a larger one is predicted in ``native_size`` tiles overlapping by half and blended with tent
        weights (centres weigh most), so the pixel scale never changes.
    memory
        For a model with ``track_memory``: the contrail probability map of the previous frame,
        ``(H, W)`` or ``(N, H, W)`` like ``composite``, in [0, 1] (usually this function's output
        for the previous frame). ``None`` (default) is an empty memory (zeros), as at the start of
        a sequence. Must be ``None`` for a model without track memory.
    tta
        Average the 8 D4 transforms. ``None`` (default) follows ``pretrained.config.tta``,
        the setting the model was evaluated with; ``True`` or ``False`` overrides it.
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
        If the composite's channel count is not ``3*(2*context_frames+1)`` (see ``composite``),
        or its scenes are smaller than ``pretrained.config.native_size``: a scene that small
        cannot be cut into ``native_size`` tiles, and resizing it to the network's input would
        change the pixel scale the model was trained on. Also if the batch is empty, ``batch_size``
        is below one, the composite is not a float array (the model expects ``[0, 1]`` floats, as
        built by ``false_color_from_spectral``), or it holds a NaN, an infinite or an out-of-range
        value: any of these turns every output NaN or gives meaningless probabilities. Also if
        ``memory`` is given for a model without track memory, does not match the composite's batch
        and size, or holds values that are not finite or outside [0, 1].
    """
    single = composite.ndim == 3
    batch = composite[None] if single else composite
    channels = 3 * (2 * pretrained.config.context_frames + 1)
    if batch.ndim != 4 or batch.shape[-1] != channels:
        msg = (
            f"expected a composite with {channels} channels "
            f"(3 x {2 * pretrained.config.context_frames + 1} frames), got shape {composite.shape}"
        )
        raise ValueError(msg)
    size = pretrained.config.native_size
    height, width = batch.shape[1], batch.shape[2]
    if height < size or width < size:
        msg = (
            f"the model was trained on {size} x {size} scenes, got {height} x {width}: a scene "
            f"smaller than the model's native size cannot be cut into native-size tiles"
        )
        raise ValueError(msg)
    if len(batch) == 0:
        raise ValueError("expected at least one composite, got an empty batch")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not np.issubdtype(batch.dtype, np.floating):
        msg = (
            f"expected a float composite in [0, 1] (as built by false_color_from_spectral), "
            f"got dtype {batch.dtype}"
        )
        raise ValueError(msg)
    if not np.isfinite(batch).all():
        raise ValueError("the composite contains NaN or infinite values")
    low, high = float(batch.min()), float(batch.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"composite values must be in [0, 1], got [{low:g}, {high:g}]")

    track_memory = pretrained.config.model.track_memory
    if memory is not None and not track_memory:
        raise ValueError("memory is given, but the model has no track memory (model.track_memory)")
    memories: np.ndarray | None = None
    if track_memory:
        if memory is None:
            memories = np.zeros(batch.shape[:3], np.float32)
        else:
            memories = memory[None] if memory.ndim == 2 else memory
            if memories.shape != batch.shape[:3]:
                msg = (
                    f"memory shape {memory.shape} does not match the composite's "
                    f"{composite.shape[:-1]}"
                )
                raise ValueError(msg)
            if not np.isfinite(memories).all() or memories.min() < 0.0 or memories.max() > 1.0:
                raise ValueError("memory values must be finite and in [0, 1]")
            memories = memories.astype(np.float32)

    if tta is None:
        tta = pretrained.config.tta
    if height == size and width == size:
        # Native scenes: unchanged path, bit-identical to predicting the composite directly.
        network = _AtInputSize(pretrained.model, (height, width)).eval()
        outputs: list[np.ndarray] = []
        for start in range(0, len(batch), batch_size):
            window = batch[start : start + batch_size]
            images = torch.stack(
                [
                    _prepare_item(
                        c,
                        None if memories is None else memories[start + j],
                        pretrained.config,
                        pixel_shift,
                    )
                    for j, c in enumerate(window)
                ]
            ).to(pretrained.device)
            if tta:
                probs = tta_predict(network, images)
            else:
                probs = torch.sigmoid(network(images)).squeeze(1)
            outputs.append(probs.float().cpu().numpy())
        result: np.ndarray[Any, Any] = np.concatenate(outputs).astype(np.float32)
        return result[0] if single else result

    # Larger scenes: cut native_size tiles overlapping by half, run each tile through the same
    # per-batch code as a native scene (same _prepare, TTA rule and batch_size batching), then
    # blend the tile probabilities with tent weights into an (H, W) map. The tiles of every
    # scene are batched together, so a batch of scenes returns a batch of blended maps.
    row_origins = tile_origins(height, size, size // 2)
    col_origins = tile_origins(width, size, size // 2)
    offsets = [(r0, c0) for r0 in row_origins for c0 in col_origins]
    tiles = [scene[r0 : r0 + size, c0 : c0 + size] for scene in batch for (r0, c0) in offsets]
    memory_tiles = (
        None
        if memories is None
        else [m[r0 : r0 + size, c0 : c0 + size] for m in memories for (r0, c0) in offsets]
    )
    network = _AtInputSize(pretrained.model, (size, size)).eval()
    tile_probs: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        images = torch.stack(
            [
                _prepare_item(
                    t,
                    None if memory_tiles is None else memory_tiles[start + j],
                    pretrained.config,
                    pixel_shift,
                )
                for j, t in enumerate(tiles[start : start + batch_size])
            ]
        ).to(pretrained.device)
        probs = tta_predict(network, images) if tta else torch.sigmoid(network(images)).squeeze(1)
        tile_probs.append(probs.float().cpu().numpy())
    all_probs = np.concatenate(tile_probs)
    weight = _blend_weights(size).astype(np.float64)
    result = np.empty((len(batch), height, width), dtype=np.float32)
    per_scene = len(offsets)
    for index in range(len(batch)):
        total = np.zeros((height, width), dtype=np.float64)
        norm = np.zeros((height, width), dtype=np.float64)
        for j, (r0, c0) in enumerate(offsets):
            prob = all_probs[index * per_scene + j]
            total[r0 : r0 + size, c0 : c0 + size] += weight * prob
            norm[r0 : r0 + size, c0 : c0 + size] += weight
        result[index] = (total / norm).astype(np.float32)
    return result[0] if single else result
