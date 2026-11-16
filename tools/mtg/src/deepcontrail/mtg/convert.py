"""Convert MTG/FCI Encord label exports and per-frame images to the DeepContrail layout.

One Encord export is a JSON file per labelled sequence (48 frames, 10 minutes apart); the
matching per-frame ``.npy`` images sit in a separate directory. This module reads both, holds
out whole calendar days as the test set (near-duplicate sequences stay together, see
:mod:`deepcontrail.mtg.split`), and writes the format v1 dataset::

    <output>/
        spectral_bands/{train,test}/{id}.npy    float32 (H, W, 3) bands, NaN = no data
        spectral_bands/bands.json               channel manifest (format v1)
        semantic_masks/{train,test}/{id}.png    uint8 union of the instances, 0 / 255 (mode "L")
        instance_masks/{train,test}/{id}.npy    int8 (N, H, W, 1), 1 inside contrail n
        metadata/dataset_info.json              source, record counts, labels, pixel shift
        metadata/scenes.json                    per-scene provenance and kept contrails
        metadata/split.json                     the seeded split, its groups and a report
        splits_train.txt / splits_test.txt      the ids written to each on-disk split
        splits_val.txt                          the val ids (they live under train/ on disk)

A frame's split is its sequence's day-group split; ``val`` scenes are written under ``train/``
on disk, so ``splits_val.txt`` is the whole-day list to feed the data config as ``val_ids``.

Every file is written atomically (temp file in the same directory then ``os.replace``), and a
frame whose three files already exist is skipped, so an interrupted run resumes; ``overwrite``
rewrites everything. Each frame is converted by a module-level worker, so the work parallelises
with a process pool.

Run it as ``deepcontrail-convert-mtg --labels ... --images ... --out ...``.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from deepcontrail.mtg.bands import band_manifest, bands_from_composite
from deepcontrail.mtg.labels import Contrail, LabelledFrame, Sequence, read_sequence
from deepcontrail.mtg.raster import block_mean, rasterize
from deepcontrail.mtg.split import day_groups, make_split, split_report

DEFAULT_TEST_FRACTION = 0.15
DEFAULT_VAL_FRACTION = 0.10
_SPLITS_ON_DISK = ("train", "test")
_LABEL_SUBDIRS = ("spectral_bands", "semantic_masks", "instance_masks")
_TIME_RE = re.compile(r"_(\d{12})_")


def _time_from_title(title: str) -> str | None:
    """Return the frame time of an ``image_title`` as an ISO string, or ``None``.

    The time is the ``_YYYYmmddHHMM_`` token in the title; a title without one has no time.
    """
    match = _TIME_RE.search(title)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M").isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Atomic writes (same idea as the OpenContrails converter, not imported)
# ---------------------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` atomically (temp file + rename)."""
    tmp = _tmp_path(path)
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def _save_image(path: Path, image: Image.Image) -> None:
    """Write ``image`` to ``path`` as a PNG atomically (temp file + rename)."""
    tmp = _tmp_path(path)
    image.save(tmp, format="PNG")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-frame worker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameJob:
    """Everything one worker needs to convert a single frame; all picklable."""

    id: str
    image_path: str
    bands_path: str
    semantic_path: str
    instance_path: str
    contrails: tuple[tuple[str, str | None, tuple[tuple[tuple[float, float], ...], ...]], ...]
    size: int
    overwrite: bool


def _convert_frame(job: _FrameJob) -> dict[str, Any]:
    """Convert one frame and return its metadata; write its three files unless they exist.

    Rasterises every labelled contrail first, so the kept-contrail list is recomputed even for
    a skipped frame; only the file writes are skipped on a resume.
    """
    size = job.size
    bands_path = Path(job.bands_path)
    semantic_path = Path(job.semantic_path)
    instance_path = Path(job.instance_path)

    contrails = tuple(
        Contrail(key=key, polygons=polygons, confidence=confidence)
        for key, confidence, polygons in job.contrails
    )

    kept_masks: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    for contrail in contrails:
        mask = rasterize(contrail.polygons, size)
        if mask.any():
            kept_masks.append(mask)
            kept.append({"key": contrail.key, "confidence": contrail.confidence})

    complete = bands_path.exists() and semantic_path.exists() and instance_path.exists()
    if complete and not job.overwrite:
        bands = np.load(bands_path)
        wrote = False
    else:
        bands = _bands_for_image(job.image_path, size, job.id)
        _save_npy(bands_path, bands)
        _save_npy(instance_path, _instance_array(kept_masks, size))
        _save_image(semantic_path, _semantic_image(kept_masks, size))
        wrote = True

    nan_fraction = float(np.isnan(bands).any(axis=-1).mean()) if bands.size else 0.0
    return {"id": job.id, "nan_fraction": nan_fraction, "kept": kept, "wrote": wrote}


def _bands_for_image(image_path: str, size: int, frame_id: str) -> np.ndarray:
    """Bands for one frame at ``size``: recover them from the composite, block-mean if needed."""
    image = np.load(image_path)
    bands = bands_from_composite(image)
    side = bands.shape[0]
    if size > side:
        msg = f"size {size} > image side {side} for {frame_id}"
        raise ValueError(msg)
    if size < side:
        if side % size:
            msg = f"image side {side} is not divisible by size {size} for {frame_id}"
            raise ValueError(msg)
        bands = block_mean(bands, side // size)
    return bands.astype(np.float32)


def _instance_array(kept_masks: list[np.ndarray], size: int) -> np.ndarray:
    """Instance masks as int8 ``(N, size, size, 1)``, one slice per kept contrail."""
    if kept_masks:
        return np.stack(kept_masks, axis=0)[..., None].astype(np.int8)
    return np.zeros((0, size, size, 1), dtype=np.int8)


def _semantic_image(kept_masks: list[np.ndarray], size: int) -> Image.Image:
    """Union of the instance masks as a uint8 ``(size, size)`` PNG (mode "L"), values 0 / 255."""
    union = np.zeros((size, size), dtype=np.uint8)
    for mask in kept_masks:
        union[mask] = 255
    return Image.fromarray(union, mode="L")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlanItem:
    """One frame to convert, with its sequence, resolved splits and worker job."""

    sequence: Sequence
    frame: LabelledFrame
    split: str
    disk_split: str
    job: _FrameJob


def convert(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    size: int = 1024,
    workers: int = 8,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert the label exports and images under ``labels_dir`` / ``images_dir`` to ``output_dir``.

    Parameters
    ----------
    labels_dir
        Directory of Encord label ``*.json`` exports, one per sequence.
    images_dir
        Directory holding the per-frame ``.npy`` images referenced by the exports.
    output_dir
        Dataset root to write in the DeepContrail format v1 layout.
    size
        Output side length; the image is block-meaned to it when larger (must divide evenly).
    workers
        Worker processes for the per-frame conversion; run inline when ``1``.
    seed
        Seed for the day-group split search.
    overwrite
        Rewrite a frame even when its three files already exist.

    Returns
    -------
    dict
        A summary with frames written and skipped, frames without an image, the kept contrail
        count, dropped empty contrails and the per-split frame fractions.

    Raises
    ------
    ValueError
        If two frames share a scene id, a group mixes projects, ``size`` is larger than an
        image side, or an image side is not divisible by ``size``.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    sequences = [read_sequence(path) for path in sorted(labels_dir.glob("*.json"))]

    n_no_image = 0
    seen: dict[str, str] = {}
    for sequence in sequences:
        for frame in sequence.frames:
            if frame.npy is None:
                n_no_image += 1
                continue
            if frame.title in seen:
                msg = (
                    f"duplicate scene id {frame.title!r} in {seen[frame.title]} and {sequence.name}"
                )
                raise ValueError(msg)
            seen[frame.title] = sequence.name

    groups = day_groups(sequences)
    assignment = make_split(
        sequences,
        groups,
        test_fraction=DEFAULT_TEST_FRACTION,
        val_fraction=DEFAULT_VAL_FRACTION,
        seed=seed,
    )

    plan: list[_PlanItem] = []
    for sequence in sequences:
        group_id = groups[sequence.name]
        split = assignment[group_id]
        disk_split = "train" if split == "val" else split
        for frame in sequence.frames:
            if frame.npy is None:
                continue
            bands_dir = output_dir / "spectral_bands" / disk_split
            png_dir = output_dir / "semantic_masks" / disk_split
            instance_dir = output_dir / "instance_masks" / disk_split
            job = _FrameJob(
                id=frame.title,
                image_path=str(images_dir / frame.npy),
                bands_path=str(bands_dir / f"{frame.title}.npy"),
                semantic_path=str(png_dir / f"{frame.title}.png"),
                instance_path=str(instance_dir / f"{frame.title}.npy"),
                contrails=tuple(
                    (c.key, c.confidence, tuple(tuple(point) for point in c.polygons))
                    for c in frame.contrails
                ),
                size=size,
                overwrite=overwrite,
            )
            plan.append(_PlanItem(sequence, frame, split, disk_split, job))

    for sub in _LABEL_SUBDIRS:
        for disk_split in _SPLITS_ON_DISK:
            (output_dir / sub / disk_split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    jobs = [item.job for item in plan]
    if workers <= 1:
        results = [_convert_frame(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_convert_frame, jobs))

    scenes: dict[str, dict[str, Any]] = {}
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    split_frames = {"train": 0, "val": 0, "test": 0}
    frames_written = frames_skipped = kept_contrails = label_contrails = 0

    for item, result in zip(plan, results, strict=True):
        kept = result["kept"]
        kept_contrails += len(kept)
        label_contrails += len(item.frame.contrails)
        split_frames[item.split] += 1
        if result["wrote"]:
            frames_written += 1
        else:
            frames_skipped += 1
        if item.split == "train":
            train_ids.append(item.frame.title)
        elif item.split == "val":
            val_ids.append(item.frame.title)
        else:
            test_ids.append(item.frame.title)
        scenes[item.frame.title] = {
            "split": item.split,
            "project": item.sequence.project,
            "sequence": item.sequence.name,
            "group": groups[item.sequence.name],
            "time": _time_from_title(item.frame.title),
            "utm_zone": item.sequence.utm_zone,
            "frame_band": item.sequence.frame_band,
            "labelled_size": item.frame.labelled_size,
            "nan_fraction": result["nan_fraction"],
            "contrails": kept,
        }

    _write_metadata(
        output_dir,
        sequences,
        groups,
        assignment,
        scenes,
        seed,
        train_ids,
        val_ids,
        test_ids,
    )

    total = len(results)
    summary: dict[str, Any] = {
        "frames_written": frames_written,
        "frames_skipped": frames_skipped,
        "frames_without_image": n_no_image,
        "contrails": kept_contrails,
        "dropped_empty_contrails": label_contrails - kept_contrails,
        "split_fractions": {
            split: round(split_frames[split] / total, 6) if total else 0.0
            for split in ("train", "val", "test")
        },
    }
    return summary


def _write_metadata(
    output_dir: Path,
    sequences: list[Sequence],
    groups: dict[str, str],
    assignment: dict[str, str],
    scenes: dict[str, dict[str, Any]],
    seed: int,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> None:
    """Write bands.json, dataset_info.json, scenes.json, split.json and the split lists."""
    n_train_dir = len(train_ids) + len(val_ids)
    n_test = len(test_ids)

    (output_dir / "spectral_bands" / "bands.json").write_text(
        json.dumps(band_manifest(), indent=2) + "\n", encoding="utf-8"
    )

    info = {
        "format_version": 1,
        "source": "MTG-I1 FCI contrail labels (EUROCONTROL, labelled in Encord)",
        "records": {"train": n_train_dir, "test": n_test},
        "labels": ["semantic", "instance"],
        "recommended_pixel_shift": 0.0,
        "temporal": False,
        "naming": "image_title",
    }
    (output_dir / "metadata" / "dataset_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    (output_dir / "metadata" / "scenes.json").write_text(
        json.dumps({id_: scenes[id_] for id_ in sorted(scenes)}, indent=2) + "\n",
        encoding="utf-8",
    )

    split_payload = {
        "seed": seed,
        "test_fraction": DEFAULT_TEST_FRACTION,
        "val_fraction": DEFAULT_VAL_FRACTION,
        "groups": {group: assignment[group] for group in sorted(assignment)},
        "report": split_report(sequences, groups, assignment),
    }
    (output_dir / "metadata" / "split.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )

    _write_split_list(output_dir / "splits_train.txt", sorted(set(train_ids) | set(val_ids)))
    _write_split_list(output_dir / "splits_test.txt", sorted(test_ids))
    _write_split_list(output_dir / "splits_val.txt", sorted(val_ids))


def _write_split_list(path: Path, ids: list[str]) -> None:
    """Write a split list: one id per line, sorted (see ``splits_*.txt``)."""
    path.write_text("".join(f"{id_}\n" for id_ in ids), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepcontrail-convert-mtg",
        description=(__doc__ or "").partition("\n")[0],
    )
    parser.add_argument(
        "--labels", type=Path, required=True, help="Directory of label *.json exports."
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="Directory of per-frame .npy images."
    )
    parser.add_argument("--out", type=Path, required=True, help="Dataset root to write.")
    parser.add_argument(
        "--size", type=int, default=1024, help="Output side length (default: 1024)."
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker processes (default: 8).")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the split search (default: 0)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rewrite frames even when their files exist."
    )
    return parser.parse_args(argv)


def _print_summary(summary: dict[str, Any]) -> None:
    frac = summary["split_fractions"]
    print(
        f"converted {summary['frames_written']} frames, skipped {summary['frames_skipped']} "
        f"(no image: {summary['frames_without_image']}); {summary['contrails']} contrails "
        f"({summary['dropped_empty_contrails']} dropped as empty at the output size)"
    )
    print(f"split fractions  train={frac['train']} val={frac['val']} test={frac['test']}")


def main(argv: list[str] | None = None) -> int:
    """Convert the exports and images to the DeepContrail layout; return 0 on success."""
    args = _parse_args(argv)
    summary = convert(
        args.labels,
        args.images,
        args.out,
        size=args.size,
        workers=args.workers,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
