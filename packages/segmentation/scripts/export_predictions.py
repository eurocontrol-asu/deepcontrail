"""Export whole-scene contrail probability maps for a pretrained bundle.

The MTG model is trained on 512 px windows of 1024 px frames, so the numbers worth
reporting must come from whole frames predicted the way users will: the library's
``predict`` cuts a scene larger than the model's native size into overlapping
native-size tiles and blends them with tent weights (the pixel scale never changes).
This script writes one float32 probability map per scene of a split; ``evaluate_scenes.py``
then scores those maps against the dataset labels. A track-memory bundle is exported in time
order, sequence by sequence, feeding each frame the map of its previous frame as its memory.

The false-colour recipe comes from the training ``--config`` (its ``data`` section), not
from the bundle: the bundle may have been trained on another sensor's band keys.

Usage
-----
::

    uv run python packages/segmentation/scripts/export_predictions.py \
        --bundle /path/to/bundle \
        --config packages/segmentation/configs/mtg.yaml \
        --split test --out runs/mtg/preds_test

    # a quick smoke test on the first ten scenes, no TTA, on the CPU
    uv run python packages/segmentation/scripts/export_predictions.py \
        --bundle hf:org/name --split test --out /tmp/preds \
        --limit 10 --no-tta --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from deepcontrail.core.contract import (
    NEIGHBOURS_PATH,
    DatasetContractError,
    context_ids,
    read_band_manifest,
    read_neighbours,
)
from deepcontrail.core.false_color import false_color_from_spectral
from deepcontrail.segmentation import (
    SegmentationDataset,
    load_pretrained,
    predict,
)
from deepcontrail.segmentation.training import load_configs

if TYPE_CHECKING:
    from deepcontrail.core.contract import BandManifest
    from deepcontrail.segmentation import SegmentationDataConfig

logger = logging.getLogger(__name__)

# Position of each split in the tuple returned by ``load_configs``.
_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def _split_dirname(split: str) -> str:
    """On-disk directory of a split's scenes: ``test/`` for the test split, ``train/`` else.

    Mirrors the dataset, where both ``train`` and ``val`` read from ``train/`` and only
    ``test`` reads from ``test/``.
    """
    return "test" if split == "test" else "train"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``path`` atomically: a temp file in the same directory, then a rename.

    A crash mid-write leaves the previous map (or nothing) in place, never a half map, so
    a rerun can resume from where it stopped.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array)
    tmp.replace(path)


def _select_data_config(
    config_path: str | Path, overrides: dict[str, Any] | None, split: str
) -> SegmentationDataConfig:
    """Return the ``SegmentationDataConfig`` of ``split`` from the training YAML.

    Index 0/1/2 of ``load_configs`` is train/val/test. ``overrides`` (e.g. ``{"data_root":
    ...}``) is forwarded to ``load_configs`` so ``--data-root`` overrides ``data.root``.
    """
    train_cfg, val_cfg, test_cfg, *_ = load_configs(config_path, overrides)
    return (train_cfg, val_cfg, test_cfg)[_SPLIT_INDEX[split]]


def _build_composite(
    stem: str,
    bands_dir: Path,
    data_cfg: SegmentationDataConfig,
    manifest: BandManifest,
    *,
    neighbours: dict[str, tuple[str | None, str | None]] | None,
    context_k: int,
) -> np.ndarray:
    """Build the false-colour composite of ``stem``, stacking context frames when ``context_k > 0``.

    With no context the composite is the scene's 3-channel false colour. With ``context_k == k`` it
    stacks the composites of the ``2k+1`` ids of :func:`deepcontrail.core.contract.context_ids`
    (the same recipe and split directory for every frame) along the channel axis into
    ``3*(2k+1)`` channels, matching how the dataset built the training images.

    Parameters
    ----------
    stem
        The centre scene id.
    bands_dir
        The split's ``spectral_bands`` directory the frames are read from.
    data_cfg
        The split's data config, for its ``false_color`` recipe.
    manifest
        The dataset's band manifest, for ``false_color_from_spectral``.
    neighbours
        Scene id -> ``(prev, next)`` links (``read_neighbours``); unused when ``context_k == 0``.
    context_k
        How many frames before and after the centre to stack (the bundle's ``context_frames``).
    """

    def _composite(scene_stem: str) -> np.ndarray:
        bands = np.load(bands_dir / f"{scene_stem}.npy")
        return false_color_from_spectral(bands, data_cfg.false_color, manifest)

    if context_k == 0:
        return _composite(stem)
    ids = context_ids(stem, neighbours, context_k)  # type: ignore[arg-type]
    return np.concatenate([_composite(cid) for cid in ids], axis=-1)


def _shape_memory(previous: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    """Return the next frame's memory: values below ``threshold`` zeroed, then times ``scale``."""
    kept = np.where(previous >= threshold, previous, 0.0) if threshold > 0 else previous
    return np.clip(kept * scale, 0.0, 1.0).astype(np.float32)


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Return ``image`` warped by ``flow`` (H, W, 2; x, y): out(y) = image(y + flow(y))."""
    from scipy.ndimage import map_coordinates

    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(np.float32)
    flow = flow.astype(np.float32)
    coords = [rows + flow[..., 1], cols + flow[..., 0]]
    return map_coordinates(image, coords, order=1, mode="constant", cval=0.0).astype(np.float32)


def _in_time_order(
    stems: list[str], neighbours: dict[str, tuple[str | None, str | None]]
) -> list[str]:
    """Order ``stems`` sequence by sequence, each in time order (``neighbours.json`` links).

    A sequence starts at a scene whose previous frame is not among ``stems`` and follows the next
    links while they stay among ``stems``. Sequences come in the order of their first scene in
    ``stems``; a scene that no sequence reaches (a broken link) keeps its order at the end.
    """
    selected = set(stems)
    ordered: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        prev = neighbours.get(stem, (None, None))[0]
        if stem in seen or (prev is not None and prev in selected):
            continue
        current: str | None = stem
        while current is not None and current in selected and current not in seen:
            ordered.append(current)
            seen.add(current)
            current = neighbours.get(current, (None, None))[1]
    ordered += [stem for stem in stems if stem not in seen]
    return ordered


def main(argv: list[str] | None = None) -> int:
    """Export one float32 probability map per scene of a split; return 0."""
    parser = argparse.ArgumentParser(
        description="Export whole-scene probability maps for a pretrained bundle."
    )
    parser.add_argument("--bundle", required=True, help="Pretrained bundle: local folder or HF id.")
    parser.add_argument("--revision", default=None, help="Hub revision when --bundle is an id.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML; its data section gives root, false_color, val_ids, pixel_shift.",
    )
    parser.add_argument(
        "--split", required=True, choices=["train", "val", "test"], help="Split to export."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the maps.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root overriding data.root of the config.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply D4 TTA (default: the bundle's config.tta).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Composites per forward pass.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, else cpu).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N scenes (smoke tests)."
    )
    parser.add_argument(
        "--memory",
        choices=["feed", "empty"],
        default="feed",
        help="Track-memory bundles: feed each map to the next frame (default), or give every "
        "frame an empty memory (the model without its memory).",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.0,
        help="Track-memory bundles: values of the fed map below this become 0 (default 0).",
    )
    parser.add_argument(
        "--memory-scale",
        type=float,
        default=1.0,
        help="Track-memory bundles: the fed map is then multiplied by this (default 1).",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run time backwards: previous and next frames swap everywhere (scene order, context "
        "frames, memory), e.g. for a backward pass to combine with a forward one.",
    )
    parser.add_argument(
        "--memory-flows",
        type=Path,
        default=None,
        help="Track-memory bundles: folder of optical flows <stem>_prev.npy (<stem>_next.npy with "
        "--reverse), (H, W, 2) with frame(y) ~ previous(y + flow(y)); the fed map is warped onto "
        "the current frame with it (no file: fed unwarped).",
    )
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = _select_data_config(args.config, {"data_root": args.data_root}, args.split)
    root = Path(data_cfg.root)
    bands_dir = root / "spectral_bands" / _split_dirname(args.split)
    manifest = read_band_manifest(root)

    pretrained = load_pretrained(args.bundle, revision=args.revision, device=device)
    stems = SegmentationDataset(data_cfg).stems
    if args.limit is not None:
        stems = stems[: args.limit]

    # A temporal bundle stacks the k context frames each side of every scene; read the links once
    # and refuse a dataset that does not carry them, so each composite matches the training recipe.
    context_k = pretrained.config.context_frames
    neighbours: dict[str, tuple[str | None, str | None]] | None = None
    if context_k > 0:
        neighbours = read_neighbours(root)
        if neighbours is None:
            msg = (
                f"{root / NEIGHBOURS_PATH}: missing; context_frames={context_k} "
                "needs metadata/neighbours.json"
            )
            raise DatasetContractError(msg)

    # Backward in time: every "previous" becomes "next" (context stacking, order, memory).
    if args.reverse and neighbours is not None:
        neighbours = {stem: (nxt, prev) for stem, (prev, nxt) in neighbours.items()}

    # A track-memory bundle feeds each scene the map of its previous frame: the scenes go in time
    # order, sequence by sequence, and each map becomes the next frame's memory.
    track_memory = pretrained.config.model.track_memory
    if track_memory:
        if neighbours is None:
            neighbours = read_neighbours(root)
            if neighbours is not None and args.reverse:
                neighbours = {stem: (nxt, prev) for stem, (prev, nxt) in neighbours.items()}
        if neighbours is None:
            msg = (
                f"{root / NEIGHBOURS_PATH}: missing; a track-memory bundle needs "
                "metadata/neighbours.json"
            )
            raise DatasetContractError(msg)
        stems = _in_time_order(stems, neighbours)
        logger.info(
            "Track-memory bundle: scenes in time order, each map is the next frame's memory"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    n = len(stems)
    tta = args.tta if args.tta is not None else pretrained.config.tta
    logger.info("Exporting %d scene(s) of the %s split of %s", n, args.split, root)

    done = 0
    last_stem: str | None = None
    last_map: np.ndarray | None = None
    for i, stem in enumerate(stems):
        out_path = args.out / f"{stem}.npy"
        if out_path.exists():  # resumable: keep the map already written
            done += 1
            if track_memory:
                last_stem, last_map = stem, np.load(out_path)
            continue
        composite = _build_composite(
            stem,
            bands_dir,
            data_cfg,
            manifest,
            neighbours=neighbours,
            context_k=context_k,
        )
        memory = None
        if (
            track_memory
            and args.memory == "feed"
            and last_map is not None
            and neighbours is not None
            and last_stem is not None
            and neighbours.get(stem, (None, None))[0] == last_stem
        ):
            memory = _shape_memory(last_map, args.memory_threshold, args.memory_scale)
            if args.memory_flows is not None:
                side = "next" if args.reverse else "prev"
                flow_path = args.memory_flows / f"{stem}_{side}.npy"
                if flow_path.exists():
                    memory = _warp(memory, np.load(flow_path))
        probs = predict(
            pretrained,
            composite,
            memory=memory,
            tta=tta,
            pixel_shift=data_cfg.pixel_shift,
            batch_size=args.batch_size,
        )
        _atomic_save_npy(out_path, np.ascontiguousarray(probs, dtype=np.float32))
        if track_memory:
            last_stem, last_map = stem, probs
        done += 1
        if i % 50 == 0:
            print(f"[{i + 1}/{n}] {stem} -> {out_path.name}")

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "bundle": str(args.bundle),
                "revision": args.revision,
                "config": str(args.config),
                "split": args.split,
                "tta": tta,
                "track_memory": track_memory,
                "memory": args.memory,
                "memory_threshold": args.memory_threshold,
                "memory_scale": args.memory_scale,
                "reverse": args.reverse,
                "memory_flows": None if args.memory_flows is None else str(args.memory_flows),
                "n_scenes": n,
                "created": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {done}/{n} scene map(s) to {args.out} (manifest written)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
