#!/usr/bin/env python
"""Convert raw OpenContrails TFRecords into the DeepContrail filesystem layout.

The raw OpenContrails export stores each sample as one TFRecord holding serialized
fields (thermal bands, per-annotator masks, the consensus pixel mask, ...). This
script decodes the shards into the layout the rest of the pipeline expects::

    <output-dir>/
        spectral_bands/{train,test}/{stem}.npy      # float32 (H, W, 10), SPECTRAL_BAND_KEYS order
        spectral_bands/bands.json                   # channel -> key, ABI band, wavelength
        images_semantic/{train,test}/{stem}.jpg     # uint8 preview, standard ash RGB
        semantic_masks/{train,test}/{stem}.png      # consensus mask, values {0, 255}
        probability_masks/{train,test}/{stem}.npy   # float32 (H, W), mean of the annotators
        annotator_masks/{train,test}/{stem}.npy     # uint8 (H, W, n_annotators)
        instance_masks/{train,test}/{stem}.npy      # int8 (N, H, W, K), 20250305 only
        splits_{train,test}.txt                     # stems with a consensus mask
        metadata/dataset_info.json                  # format, counts, recommended pixel_shift

Each sample is named after its TFRecord ``record_id`` (``{stem}`` above), so a name
does not depend on shard order: any subset of shards (``--file-glob``) converts to
the same names as a full run.

* Raw ``train`` shards go to ``train/``. Validation is drawn from ``train/`` at
  load time (``DataConfig.val_fraction`` or ``fold``), so no ``val/`` folder exists.
* Raw ``validation`` shards (Kaggle's validation set) go to ``test/``.

No composite is stored as ``.npy``: training builds it at load time from
``spectral_bands/`` with the bands configured in ``DataConfig.false_color``.

Two export flavours are handled transparently (auto-detected from the first record):

* **20250305 (single-frame):** bands are ``(H, W, 1)``; instance masks are present.
* **20230419 (temporal clips):** bands are ``(H, W, 8)``: ``n_times_before`` frames,
  the labelled frame, then ``n_times_after`` frames. Every frame is written; only the
  labelled frame (plain ``{stem}``) has masks. Context frames are named by their
  distance to it: ``{stem}_b4`` … ``{stem}_b1`` and ``{stem}_a1`` … ``{stem}_a3``.
  No instance masks.

Every file is written atomically (temp file + rename), so an existing file is always
complete. By default a record is skipped when its consensus mask and every frame's
spectral bands and preview exist, which makes an interrupted run resumable;
``--no-skip-existing`` rewrites everything.

TensorFlow (``uv sync --extra convert``) is imported lazily, only when shards are
decoded, so this module can be imported next to torch (e.g. in the test suite).
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

_SPLITS = ("train", "test")


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
# Naming
# ---------------------------------------------------------------------------


def _record_stem(record_id: int | None) -> str:
    """Name a sample after its TFRecord ``record_id``.

    Raises
    ------
    ValueError
        If the record has no ``record_id``: there is no order-independent name.
    """
    if record_id is None:
        msg = "TFRecord has no record_id; cannot name the sample"
        raise ValueError(msg)
    return str(record_id)


def _frame_stem(base_stem: str, t: int, label_idx: int, n_frames: int) -> str:
    """Compute the output filename stem for frame ``t`` within a clip."""
    if n_frames == 1 or t == label_idx:
        return base_stem
    if t < label_idx:
        return f"{base_stem}_b{label_idx - t}"
    return f"{base_stem}_a{t - label_idx}"


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


# ---------------------------------------------------------------------------
# Mask processing
# ---------------------------------------------------------------------------


def _probability_mask(individual: np.ndarray) -> np.ndarray:
    """Consensus probability mask = mean of the annotators' binary masks.

    ``individual`` has shape ``(H, W, 1, A)``; the result is ``(H, W)`` float32.
    """
    return (
        individual.reshape(individual.shape[0], individual.shape[1], -1)
        .mean(axis=-1)
        .astype(np.float32)
    )


def _annotator_masks(individual: np.ndarray) -> np.ndarray:
    """Return the per-annotator binary masks as ``(H, W, A)`` uint8."""
    return individual.reshape(individual.shape[0], individual.shape[1], -1).astype(np.uint8)


def _instance_masks(instance: np.ndarray) -> np.ndarray:
    """Return instance masks as ``(N, H, W, K)`` int8.

    Source shape is ``(N, H, W, 1, K)``; the trailing ``(1, K)`` collapses to a
    single ``K`` dim whose width varies per sample. Records with no annotated
    instances have ``N == 0`` and are emitted as an empty ``(0, H, W, K)`` array.
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


def _verify_consensus(png_path: Path, pixel_mask: np.ndarray, stem: str) -> None:
    """Raise if the record's consensus mask differs from an existing PNG.

    Used when re-converting into an existing dataset: a mismatch means the data
    on disk and the TFRecords disagree, so fail loudly rather than overwrite.
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
# Writing and resume
# ---------------------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` atomically (temp file + rename)."""
    tmp = _tmp_path(path)
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def _save_image(path: Path, image: Image.Image, fmt: str) -> None:
    """Write ``image`` to ``path`` in ``fmt`` (``"PNG"``/``"JPEG"``) atomically."""
    tmp = _tmp_path(path)
    image.save(tmp, format=fmt)
    tmp.replace(path)


def _record_complete(output_dir: Path, split: str, stem: str, frame_stems: list[str]) -> bool:
    """Whether a record's consensus mask and every frame's bands and preview exist."""
    if not (output_dir / "semantic_masks" / split / f"{stem}.png").exists():
        return False
    return all(
        (output_dir / "spectral_bands" / split / f"{f}.npy").exists()
        and (output_dir / "images_semantic" / split / f"{f}.jpg").exists()
        for f in frame_stems
    )


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


def _process_shard(
    shard_path: Path,
    output_dir: Path,
    *,
    fmt: FormatInfo,
    split: str,
    skip_existing: bool,
    verify: bool,
) -> tuple[int, int]:
    """Decode one shard into ``split``; return ``(processed, skipped)`` record counts."""
    import tensorflow as tf

    processed = 0
    skipped = 0
    for raw in tf.data.TFRecordDataset(str(shard_path)):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        stem = _record_stem(_scalar_int(example, "record_id"))

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
        frame_stems = [_frame_stem(stem, t, label_frame, n_frames) for t in range(n_frames)]

        if skip_existing and _record_complete(output_dir, split, stem, frame_stems):
            skipped += 1
            continue

        bands = _decode_bands(example)
        pixel_mask = _decode_mask(example, "human_pixel_masks")
        individual = _decode_mask(example, "human_individual_masks")

        if pixel_mask is not None:
            mask_path = output_dir / "semantic_masks" / split / f"{stem}.png"
            if verify and mask_path.exists():
                _verify_consensus(mask_path, pixel_mask, stem)
            _save_image(
                mask_path, Image.fromarray(_consensus_mask_png(pixel_mask), mode="L"), "PNG"
            )

        if individual is not None:
            _save_npy(
                output_dir / "probability_masks" / split / f"{stem}.npy",
                _probability_mask(individual),
            )
            _save_npy(
                output_dir / "annotator_masks" / split / f"{stem}.npy",
                _annotator_masks(individual),
            )

        if fmt.has_instance_masks:
            instance = _decode_mask(example, "human_individual_instance_masks")
            if instance is not None:
                _save_npy(
                    output_dir / "instance_masks" / split / f"{stem}.npy",
                    _instance_masks(instance),
                )

        for t, fstem in enumerate(frame_stems):
            spectral = _spectral_bands_frame(bands, t)
            _save_npy(output_dir / "spectral_bands" / split / f"{fstem}.npy", spectral)
            _save_image(
                output_dir / "images_semantic" / split / f"{fstem}.jpg",
                Image.fromarray(_preview_rgb(spectral), mode="RGB"),
                "JPEG",
            )

        processed += 1

    return processed, skipped


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


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
        for split in _SPLITS:
            (output_dir / sub / split).mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "spectral_bands" / "bands.json"
    manifest.write_text(json.dumps(_band_manifest(), indent=2) + "\n")


def _write_split_lists(output_dir: Path) -> None:
    """Write ``splits_{split}.txt`` from the consensus masks on disk (complete after resumes)."""
    for split in _SPLITS:
        stems = sorted(p.stem for p in (output_dir / "semantic_masks" / split).glob("*.png"))
        (output_dir / f"splits_{split}.txt").write_text("".join(f"{s}\n" for s in stems))


# Measured on the two OpenContrails exports: same images, but 20250305 masks are
# rasterised ~0.5 px up-left of the 20230419 (Kaggle-convention) ones.
_KNOWN_EXPORTS: dict[tuple[bool, bool], tuple[str, float]] = {
    (True, False): ("20230419", 0.5),
    (False, True): ("20250305", 0.0),
}


def _dataset_info(fmt: FormatInfo, records: dict[str, int]) -> dict[str, Any]:
    """Describe a converted dataset, including the ``pixel_shift`` to train it with."""
    export, shift = _KNOWN_EXPORTS.get((fmt.is_temporal, fmt.has_instance_masks), (None, None))
    note = (
        "Use as DataConfig.pixel_shift. 20230419 masks follow the Kaggle convention (0.5); "
        "20250305 masks are already aligned with the images (0.0)."
        if export
        else "Unknown export: measure the image/mask alignment before choosing pixel_shift."
    )
    return {
        "source": "OpenContrails GOES-16 ABI TFRecords",
        "export": export,
        "temporal": fmt.is_temporal,
        "instance_masks": fmt.has_instance_masks,
        "naming": "record_id",
        "records": records,
        "recommended_pixel_shift": shift,
        "pixel_shift_note": note,
    }


def _write_dataset_info(output_dir: Path, fmt: FormatInfo) -> None:
    """Write ``metadata/dataset_info.json``, counting records from ``splits_*.txt``."""
    records = {s: len((output_dir / f"splits_{s}.txt").read_text().split()) for s in _SPLITS}
    path = output_dir / "metadata" / "dataset_info.json"
    path.write_text(json.dumps(_dataset_info(fmt, records), indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tfrecord-dir",
        type=Path,
        required=True,
        help="Directory holding ``train.tfrecords-*`` and ``validation.tfrecords-*`` shards.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset root to write.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records whose mask and every frame's bands and preview already exist, "
        "so an interrupted run resumes (default). --no-skip-existing rewrites them.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=True,
        help="Skip the pixel-identical check of consensus masks against existing PNGs.",
    )
    parser.add_argument(
        "--file-glob",
        default=None,
        help="Only process shards whose basename matches this fnmatch glob "
        '(e.g. "*.tfrecords-00000-of-00100"). Names do not depend on shard order, '
        "so a subset converts to the same files as a full run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Convert the TFRecord shards to the filesystem layout and return 0."""
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    tf_dir = Path(args.tfrecord_dir)

    _prepare_output_dir(output_dir)

    shards_by_split = {
        "train": _list_shards(tf_dir, "train.tfrecords", args.file_glob),
        "test": _list_shards(tf_dir, "validation.tfrecords", args.file_glob),
    }
    all_shards = shards_by_split["train"] + shards_by_split["test"]
    if not all_shards:
        print("No shards found.")
        return 0

    fmt = _probe_format(all_shards)
    label = "temporal (multi-frame)" if fmt.is_temporal else "single-frame"
    print(f"Detected format: {label}, instance_masks={'yes' if fmt.has_instance_masks else 'no'}")
    print(
        f"Processing {len(shards_by_split['train'])} train + "
        f"{len(shards_by_split['test'])} validation shard(s)"
    )

    done = 0
    for split, shards in shards_by_split.items():
        for shard in shards:
            done += 1
            proc, skipped = _process_shard(
                shard,
                output_dir,
                fmt=fmt,
                split=split,
                skip_existing=args.skip_existing,
                verify=args.verify,
            )
            print(
                f"[shard {done}/{len(all_shards)}] {split}: "
                f"processed {proc} records ({skipped} skipped)",
                flush=True,
            )

    _write_split_lists(output_dir)
    _write_dataset_info(output_dir, fmt)
    info = json.loads((output_dir / "metadata" / "dataset_info.json").read_text())
    print(
        f"Dataset now holds {info['records']['train']} train + {info['records']['test']} "
        f"test records; recommended pixel_shift: {info['recommended_pixel_shift']}"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
