#!/usr/bin/env python
"""Convert raw OpenContrails TFRecords into the DeepContrail filesystem layout.

The raw OpenContrails export stores each sample as one TFRecord holding a set of
serialized string fields (thermal bands, per-annotator instance masks, the
consensus pixel mask, ...). This script decodes those shards into the layout the
rest of the pipeline expects::

    <output-dir>/
        images_semantic/{train,val,test}/{stem}.jpg        # uint8 preview, standard ash RGB
        semantic_masks/{train,val,test}/{stem}.png         # (already present)
        probability_masks/{train,val,test}/{stem}.npy      # float32 (H, W)
        annotator_masks/{train,val,test}/{stem}.npy        # uint8   (H, W, 4)
        instance_masks/{train,val,test}/{stem}.npy         # int8    (N, H, W, 4)
        spectral_bands/{train,val,test}/{stem}.npy         # float32 (H, W, 10)
        spectral_bands/bands.json                          # channel -> key, ABI band, wavelength
        metadata/{train,val,test}_record_ids.json          # {stem: record_id}

``spectral_bands`` channels follow ``deepcontrail.data.SPECTRAL_BAND_KEYS``. No
composite is stored as ``.npy``: training builds it at load time from the
configured bands (``DataConfig.false_color``).

Two export flavours are handled transparently:

* **20250305 (single-frame):** each band is ``(H, W, 1)``; instance masks are
  present. The trailing dimension is squeezed away.
* **20230419 (temporal clips):** each band is ``(H, W, 8)`` — an 8-frame clip
  with ``n_times_before`` frames before the labelled frame, 1 labelled, and
  ``n_times_after`` after. All 8 frames are extracted.  Only the labelled frame
  (at index ``n_times_before``) carries masks; context frames are imagery only.
  Instance masks are absent in this export.

Context frames use distance-based suffixes relative to the labelled frame:
``{stem}_b4`` … ``{stem}_b1`` (before) and ``{stem}_a1`` … ``{stem}_a3``
(after).  The labelled frame keeps the plain stem so the existing dataset loader
finds it without modification.

The format is auto-detected by probing the first record.

The record→stem mapping is positional and driven by the dataset's own split
files (kept in ``output_dir``), so the original sample numbers are preserved
exactly:

* Raw ``train`` shards, processed in sorted order, number records by their
  global position ``p`` (1..20,088) as ``train_{p:06d}``. A record whose stem
  is listed in ``splits_val.txt`` is held out as a ``val`` sample; every other
  record is a ``train`` sample.  The held-out samples keep the ``train_`` prefix
  even though they live under ``semantic_masks/val/``.
* Raw ``validation`` shards number records ``q`` (1..1,879) as ``val_{q:06d}``
  and become the ``test`` split.

Because the stems come from the existing dataset, each converted sample can be
verified against ``semantic_masks/{split}/{stem}.png`` pixel-for-pixel.

Only ``tensorflow`` (already installed in the project venv) is imported, and it
is imported lazily so this script can be linted/run in environments where the
shipped library must stay torch-free. This module lives under ``scripts/`` — it
is tooling, not part of ``src/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from deepcontrail.data._loading import false_color_from_spectral
from deepcontrail.data.config import (
    SPECTRAL_BAND_INFO,
    SPECTRAL_BAND_KEYS,
    FalseColorChannel,
    FalseColorConfig,
)

if TYPE_CHECKING:
    import tensorflow as tf

# Silence TensorFlow's CUDA/GPU chatter; this is a CPU-bound conversion.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclasses.dataclass(frozen=True)
class FormatInfo:
    """Auto-detected TFRecord format metadata."""

    is_temporal: bool
    has_instance_masks: bool


# ---------------------------------------------------------------------------
# Protobuf field extraction
# ---------------------------------------------------------------------------


def _feature_bytes(example: tf.train.Example, key: str) -> bytes | None:
    """Return the first serialized ``bytes_list`` value for ``key``, or ``None``."""
    fv = example.features.feature.get(key, None)
    if fv is None or fv.WhichOneof("kind") != "bytes_list":
        return None
    values = list(fv.bytes_list.value)
    return values[0] if values else None


def _scalar_int(example: tf.train.Example, key: str) -> int | None:
    """Return the first ``int64_list`` scalar for ``key``, or ``None``."""
    fv = example.features.feature.get(key, None)
    if fv is None or fv.WhichOneof("kind") != "int64_list":
        return None
    values = list(fv.int64_list.value)
    return int(values[0]) if values else None


def _decode_mask(example: tf.train.Example, key: str) -> np.ndarray | None:
    """Decode a serialized int32 mask tensor, or ``None`` if the field is absent."""
    import tensorflow as tf

    raw = _feature_bytes(example, key)
    if raw is None:
        return None
    return tf.io.parse_tensor(raw, out_type=tf.int32).numpy()


def _decode_bands(example: tf.train.Example) -> dict[str, np.ndarray]:
    """Decode all spectral band tensors into ``(H, W, T)`` float64 arrays."""
    import tensorflow as tf

    decoded: dict[str, np.ndarray] = {}
    for key in SPECTRAL_BAND_KEYS:
        raw = _feature_bytes(example, key)
        if raw is not None:
            decoded[key] = tf.io.parse_tensor(raw, out_type=tf.float64).numpy()
    return decoded


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------


def _band_frame(bands: dict[str, np.ndarray], key: str, t: int) -> np.ndarray:
    """Return band ``key`` at frame ``t`` as a 2-D ``(H, W)`` float64 array."""
    return bands[key][:, :, t]


# Standard ash RGB (OpenContrails paper): G = band14 - band11, i.e. data_11um - cloud_top.
_PREVIEW_FALSE_COLOR = FalseColorConfig(
    channels=(
        FalseColorChannel(bands=("data_12um", "data_11um"), bounds=(-4.0, 2.0)),
        FalseColorChannel(bands=("data_11um", "cloud_top"), bounds=(-4.0, 5.0)),
        FalseColorChannel(bands=("data_11um",), bounds=(243.0, 303.0)),
    )
)


def _preview_rgb(spectral: np.ndarray) -> np.ndarray:
    """Render a ``(H, W, 10)`` spectral frame as a uint8 standard ash RGB preview.

    Written to ``images_semantic/``. Spectral training does not use these; it
    builds its composite at load time from ``spectral_bands/`` with the bands
    set in ``DataConfig.false_color``.
    """
    return (false_color_from_spectral(spectral, _PREVIEW_FALSE_COLOR) * 255).astype(np.uint8)


def _spectral_bands_frame(bands: dict[str, np.ndarray], t: int) -> np.ndarray:
    """Stack all 10 thermal bands at frame ``t`` into ``(H, W, 10)`` float32."""
    return np.stack([_band_frame(bands, key, t) for key in SPECTRAL_BAND_KEYS], axis=-1).astype(
        np.float32
    )


def _frame_stem(base_stem: str, t: int, label_idx: int, n_frames: int) -> str:
    """Compute the output filename stem for frame ``t`` within a clip."""
    if n_frames == 1 or t == label_idx:
        return base_stem
    if t < label_idx:
        return f"{base_stem}_b{label_idx - t}"
    return f"{base_stem}_a{t - label_idx}"


# ---------------------------------------------------------------------------
# Mask processing (unchanged between formats)
# ---------------------------------------------------------------------------


def _probability_mask(individual: np.ndarray) -> np.ndarray:
    """Consensus probability mask = mean of the four annotators' binary masks.

    ``individual`` has shape ``(H, W, 1, 4)``; the result is ``(H, W)`` float32
    with values in ``{0, 0.25, 0.5, 0.75, 1.0}``.
    """
    return (
        individual.reshape(individual.shape[0], individual.shape[1], -1)
        .mean(axis=-1)
        .astype(np.float32)
    )


def _annotator_masks(individual: np.ndarray) -> np.ndarray:
    """Return the four per-annotator binary masks as ``(H, W, 4)`` uint8."""
    return individual.reshape(individual.shape[0], individual.shape[1], -1).astype(np.uint8)


def _instance_masks(instance: np.ndarray) -> np.ndarray:
    """Return instance masks as ``(N, H, W, K)`` int8.

    Source shape is ``(N, H, W, 1, K)``; the trailing ``(1, K)`` collapses to a
    single ``K`` dim whose width varies per sample (4-7 annotators here). Records
    with no annotated instances have ``N == 0`` and are emitted as an empty
    ``(0, H, W, K)`` array.
    """
    lead = list(instance.shape[:3])
    trailing = 1
    for dim in instance.shape[3:]:
        trailing *= dim
    return instance.reshape(*tuple(lead), trailing).astype(np.int8)


def _consensus_mask_png(pixel_mask: np.ndarray) -> np.ndarray:
    """Binary consensus mask as a ``(H, W)`` uint8 array with values {0, 255}."""
    mask = pixel_mask.reshape(pixel_mask.shape[0], pixel_mask.shape[1])
    return (mask * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------


def _verify_consensus(png_path: Path, pixel_mask: np.ndarray, stem: str) -> None:
    """Assert the record's consensus mask matches the existing PNG exactly.

    The PNG is binarized (``> 0`` → 1) to match the record's ``{0, 1}``
    convention. A mismatch means the record→stem mapping is wrong, so we fail
    loudly rather than silently corrupt the dataset.
    """
    with Image.open(png_path) as img:
        png = np.array(img.convert("L"), dtype=np.uint8)
    binary = (png > 0).astype(np.uint8)
    consensus = pixel_mask.reshape(pixel_mask.shape[0], pixel_mask.shape[1])

    if binary.shape != consensus.shape:
        msg = f"{stem}: shape mismatch png {binary.shape} vs consensus {consensus.shape}"
        raise ValueError(msg)

    diffs = int(np.sum(binary != consensus))
    if diffs:
        msg = f"{stem}: verification failed — {diffs} differing pixels"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _probe_format(shards: list[Path]) -> FormatInfo:
    """Auto-detect the TFRecord format by examining the first available record."""
    import tensorflow as tf

    for shard in shards:
        for raw in tf.data.TFRecordDataset(str(shard)):
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            band_raw = _feature_bytes(example, "data_11um")
            if band_raw is not None:
                band = tf.io.parse_tensor(band_raw, out_type=tf.float64).numpy()
                is_temporal = band.shape[2] > 1
            else:
                is_temporal = False
            has_instance = _feature_bytes(example, "human_individual_instance_masks") is not None
            return FormatInfo(is_temporal=is_temporal, has_instance_masks=has_instance)
    msg = "No records found in any shard"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------


def _list_shards(tf_dir: Path, prefix: str, file_glob: str | None) -> list[Path]:
    """Sorted shard paths in ``tf_dir`` matching ``prefix`` and optional glob."""
    candidates = sorted(p for p in tf_dir.glob(f"{prefix}-*") if p.is_file())
    if file_glob is None:
        return candidates
    return [p for p in candidates if fnmatch.fnmatch(p.name, file_glob)]


def _load_stems(path: Path) -> list[str]:
    """Read a ``splits_*.txt`` file into a list of stem names (skipping blanks)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _process_shard(
    shard_path: Path,
    output_dir: Path,
    *,
    fmt: FormatInfo,
    start_pos: int,
    is_validation: bool,
    val_stems: set[str] | None,
    metadata: dict[str, dict[str, int]],
    skip_existing: bool,
    verify: bool,
) -> tuple[int, int]:
    """Decode one shard; return ``(processed, skipped)`` record counts.

    ``start_pos`` is the 0-based global position of the shard's first record; it
    advances past skipped records so stems stay aligned across shards.
    """
    import tensorflow as tf

    processed = 0
    skipped = 0
    pos = start_pos
    for raw in tf.data.TFRecordDataset(str(shard_path)):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())

        record_id = _scalar_int(example, "record_id") or pos
        if is_validation:
            stem = f"val_{pos + 1:06d}"
            split = "test"
        else:
            stem = f"train_{pos + 1:06d}"
            split = "val" if (val_stems is not None and stem in val_stems) else "train"
        pos += 1

        if fmt.is_temporal:
            nb_raw = _scalar_int(example, "n_times_before")
            nt_raw = _scalar_int(example, "n_times")
            na_raw = _scalar_int(example, "n_times_after")
            nb = nb_raw if nb_raw is not None else 4
            nt = nt_raw if nt_raw is not None else 1
            na = na_raw if na_raw is not None else 3
            n_frames = nb + nt + na
            label_frame = nb
        else:
            n_frames = 1
            label_frame = 0

        mask_path = output_dir / "semantic_masks" / split / f"{stem}.png"
        sb_path = output_dir / "spectral_bands" / split / f"{stem}.npy"
        if skip_existing and sb_path.exists() and mask_path.exists():
            metadata[split].setdefault(stem, record_id)
            skipped += 1
            continue

        bands = _decode_bands(example)

        pixel_mask = _decode_mask(example, "human_pixel_masks")
        individual = _decode_mask(example, "human_individual_masks")

        if pixel_mask is not None:
            if verify:
                existing_png = output_dir / "semantic_masks" / split / f"{stem}.png"
                if existing_png.exists():
                    _verify_consensus(existing_png, pixel_mask, stem)

            Image.fromarray(_consensus_mask_png(pixel_mask), mode="L").save(
                output_dir / "semantic_masks" / split / f"{stem}.png"
            )

        if individual is not None:
            np.save(
                output_dir / "probability_masks" / split / f"{stem}.npy",
                _probability_mask(individual),
            )
            np.save(
                output_dir / "annotator_masks" / split / f"{stem}.npy",
                _annotator_masks(individual),
            )

        if fmt.has_instance_masks:
            instance = _decode_mask(example, "human_individual_instance_masks")
            if instance is not None:
                np.save(
                    output_dir / "instance_masks" / split / f"{stem}.npy",
                    _instance_masks(instance),
                )

        for t in range(n_frames):
            fstem = _frame_stem(stem, t, label_frame, n_frames)

            spectral = _spectral_bands_frame(bands, t)
            np.save(output_dir / "spectral_bands" / split / f"{fstem}.npy", spectral)
            Image.fromarray(_preview_rgb(spectral), mode="RGB").save(
                output_dir / "images_semantic" / split / f"{fstem}.jpg"
            )

        metadata[split][stem] = record_id
        processed += 1

    return processed, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfrecord-dir",
        type=Path,
        required=True,
        help="Directory holding ``train.tfrecords-*`` and ``validation.tfrecords-*`` shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dataset root; ``semantic_masks/`` must already live here (used by --verify).",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Resumable: skip a sample when its probability mask already exists (default).",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=True,
        help="Skip the pixel-identical consensus check against existing PNG masks.",
    )
    parser.add_argument(
        "--file-glob",
        default=None,
        help="Only process shards whose basename matches this fnmatch glob "
        '(e.g. "*.tfrecords-00000-of-00100") — handy for a smoke test on one shard.',
    )
    return parser.parse_args(argv)


_OUTPUT_SUBDIRS = (
    "images_semantic",
    "semantic_masks",
    "probability_masks",
    "annotator_masks",
    "instance_masks",
    "spectral_bands",
)


def _band_manifest() -> dict[str, Any]:
    """Describe the channels of ``spectral_bands/*.npy`` for readers without this code."""
    channels = []
    for index, key in enumerate(SPECTRAL_BAND_KEYS):
        info = SPECTRAL_BAND_INFO[key]
        channels.append(
            {
                "index": index,
                "key": key,
                "abi_band": info.abi_band,
                "wavelength_um": info.wavelength_um,
                "name": info.name,
                "derived_from": list(info.derived_from) if info.derived_from else None,
            }
        )
    return {
        "description": "Channel layout of spectral_bands/{split}/{stem}.npy (GOES-16 ABI)",
        "shape": ["H", "W", len(SPECTRAL_BAND_KEYS)],
        "dtype": "float32",
        "units": "K",
        "channels": channels,
    }


def _prepare_output_dir(output_dir: Path) -> None:
    """Create the output layout and write ``spectral_bands/bands.json``."""
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    for sub in _OUTPUT_SUBDIRS:
        for split in ("train", "val", "test"):
            (output_dir / sub / split).mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "spectral_bands" / "bands.json"
    manifest.write_text(json.dumps(_band_manifest(), indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Convert the TFRecord shards to the filesystem layout and return 0."""
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    tf_dir = Path(args.tfrecord_dir)

    _prepare_output_dir(output_dir)

    # Resume: keep record_ids from a prior interrupted run so skipped stems are
    # retained in the metadata output.
    metadata: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for split in metadata:
        meta_path = output_dir / "metadata" / f"{split}_record_ids.json"
        if meta_path.exists():
            metadata[split] = json.loads(meta_path.read_text())

    # The held-out (val) stems live under ``semantic_masks/val/`` but reuse the
    # ``train_`` prefix, so classify raw-train records by membership here.
    val_stems_path = output_dir / "splits_val.txt"
    val_stems: set[str] | None = (
        set(_load_stems(val_stems_path)) if val_stems_path.exists() else None
    )

    train_files = _list_shards(tf_dir, "train.tfrecords", args.file_glob)
    val_files = _list_shards(tf_dir, "validation.tfrecords", args.file_glob)
    total = len(train_files) + len(val_files)
    all_shards = train_files + val_files
    if not all_shards:
        print("No shards found.")
        return 0

    fmt = _probe_format(all_shards)
    label = "temporal (multi-frame)" if fmt.is_temporal else "single-frame"
    print(f"Detected format: {label}, instance_masks={'yes' if fmt.has_instance_masks else 'no'}")
    print(f"Processing {len(train_files)} train + {len(val_files)} validation shard(s)")

    shard_index = 0
    train_start = 0
    for shard in train_files:
        shard_index += 1
        proc, skipped = _process_shard(
            shard,
            output_dir,
            fmt=fmt,
            start_pos=train_start,
            is_validation=False,
            val_stems=val_stems,
            metadata=metadata,
            skip_existing=args.skip_existing,
            verify=args.verify,
        )
        train_start += proc + skipped
        print(f"[shard {shard_index}/{total}] processed {proc} records ({skipped} skipped)")

    val_start = 0
    for shard in val_files:
        shard_index += 1
        proc, skipped = _process_shard(
            shard,
            output_dir,
            fmt=fmt,
            start_pos=val_start,
            is_validation=True,
            val_stems=None,
            metadata=metadata,
            skip_existing=args.skip_existing,
            verify=args.verify,
        )
        val_start += proc + skipped
        print(f"[shard {shard_index}/{total}] processed {proc} records ({skipped} skipped)")

    print(
        f"Converted {len(metadata['train'])} train + {len(metadata['val'])} val + "
        f"{len(metadata['test'])} test records"
    )

    for split in metadata:
        if metadata[split]:
            meta_path = output_dir / "metadata" / f"{split}_record_ids.json"
            with meta_path.open("w") as f:
                json.dump(
                    {stem: metadata[split][stem] for stem in sorted(metadata[split])},
                    f,
                    indent=2,
                )

    for split in metadata:
        if metadata[split]:
            splits_path = output_dir / f"splits_{split}.txt"
            with splits_path.open("w") as f:
                for stem in sorted(metadata[split]):
                    f.write(f"{stem}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
