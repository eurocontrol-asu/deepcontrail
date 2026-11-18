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

**Several labellers.** When the label folder holds one JSON per (sequence, labeller) — the
same ``sequence`` name once per labeller, each ``project_name`` ending in a different ``_L<k>``
and the same frame list — the labellers are combined at the *pixel* level (contrail ids are not
comparable across people). For every frame the converter writes one union mask per labeller
(``annotator_masks`` uint8 ``(H, W, A)`` in the sorted labeller order), their mean
(``probability_masks`` float32 ``(H, W)``, the labeller agreement) and the majority vote
(``semantic_masks`` PNG, a pixel is drawn when ``votes * 2 > A``). Instance masks are not
supported with several labellers. ``dataset_info`` then declares
``["semantic", "probability", "annotator"]`` and each ``scenes.json`` entry carries
``contrails_by_annotator`` and ``annotators``.

Two options reuse an existing dataset: ``link_bands_from`` hard-links the band files of a
converted dataset instead of computing them from the images, and ``split_from`` reuses that
dataset's frozen ``metadata/split.json`` assignment instead of searching a new one.

A frame's split is its sequence's day-group split; ``val`` scenes are written under ``train/``
on disk, so ``splits_val.txt`` is the whole-day list to feed the data config as ``val_ids``.

Every file is written atomically (temp file in the same directory then ``os.replace``), and a
frame whose files already exist is skipped, so an interrupted run resumes; ``overwrite``
rewrites everything. Each frame is converted by a module-level worker, so the work parallelises
with a process pool.

Run it as ``deepcontrail-convert-mtg --labels ... --images ... --out ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from deepcontrail.core.contract import DatasetInfo
from deepcontrail.mtg.bands import band_manifest, bands_from_composite
from deepcontrail.mtg.labels import (
    Contrail,
    LabelledFrame,
    Sequence,
    annotator_of,
    read_sequence,
)
from deepcontrail.mtg.raster import block_mean, rasterize
from deepcontrail.mtg.split import day_groups, make_split, split_report

DEFAULT_TEST_FRACTION = 0.15
DEFAULT_VAL_FRACTION = 0.10
_SPLITS_ON_DISK = ("train", "test")
_TIME_RE = re.compile(r"_(\d{12})_")

# Nested tuple shapes carried through the picklable worker job.
_ContrailTuple = tuple[str, str | None, tuple[tuple[tuple[float, float], ...], ...]]
_AnnotatorContrails = tuple[_ContrailTuple, ...]
_FrameAnnotators = tuple[_AnnotatorContrails, ...]


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
    instance_path: str | None
    annotator_path: str | None
    probability_path: str | None
    bands_source: str | None
    annotators_contrails: _FrameAnnotators
    size: int
    overwrite: bool


def _convert_frame(job: _FrameJob) -> dict[str, Any]:
    """Convert one frame and return its metadata; write its files unless they exist.

    Rasterises every labelled contrail first, so the kept-contrail list is recomputed even for
    a skipped frame; only the file writes are skipped on a resume.
    """
    size = job.size
    bands_path = Path(job.bands_path)
    semantic_path = Path(job.semantic_path)
    instance_path = Path(job.instance_path) if job.instance_path else None
    annotator_path = Path(job.annotator_path) if job.annotator_path else None
    probability_path = Path(job.probability_path) if job.probability_path else None

    per_annotator_masks: list[list[np.ndarray]] = []
    per_annotator_kept: list[list[dict[str, Any]]] = []
    for annotator_contrails in job.annotators_contrails:
        contrails = tuple(
            Contrail(key=key, polygons=polygons, confidence=confidence)
            for key, confidence, polygons in annotator_contrails
        )
        masks: list[np.ndarray] = []
        kept: list[dict[str, Any]] = []
        for contrail in contrails:
            mask = rasterize(contrail.polygons, size)
            if mask.any():
                masks.append(mask)
                kept.append({"key": contrail.key, "confidence": contrail.confidence})
        per_annotator_masks.append(masks)
        per_annotator_kept.append(kept)

    output_files: list[Path] = [bands_path, semantic_path]
    if instance_path is not None:
        output_files.append(instance_path)
    if annotator_path is not None and probability_path is not None:
        output_files.append(annotator_path)
        output_files.append(probability_path)
    complete = all(path.exists() for path in output_files) and not job.overwrite

    if not complete:
        if job.bands_source is not None:
            _hard_link_bands(job.bands_source, bands_path, job.id)
        else:
            _save_npy(bands_path, _bands_for_image(job.image_path, size, job.id))
        _write_label_masks(
            per_annotator_masks,
            size,
            semantic_path=semantic_path,
            instance_path=instance_path,
            annotator_path=annotator_path,
            probability_path=probability_path,
        )
        wrote = True
    else:
        wrote = False

    bands = np.load(bands_path)
    nan_fraction = float(np.isnan(bands).any(axis=-1).mean()) if bands.size else 0.0
    return {
        "id": job.id,
        "nan_fraction": nan_fraction,
        "wrote": wrote,
        "kept_by_annotator": per_annotator_kept,
    }


def _hard_link_bands(source: str, dest: Path, frame_id: str) -> None:
    """Hard-link a band file from a linked dataset to ``dest``.

    Raises
    ------
    ValueError
        If the source band file does not exist.
    """
    src = Path(source)
    if not src.is_file():
        msg = f"band file missing for {frame_id} in the linked dataset: {src}"
        raise ValueError(msg)
    with suppress(FileExistsError):  # a concurrent or resumed write already made the link
        os.link(src, dest)


def _union_of(masks: list[np.ndarray], size: int) -> np.ndarray:
    """Union (any covered) of per-contrail boolean masks as a ``(size, size)`` bool array."""
    union = np.zeros((size, size), dtype=bool)
    for mask in masks:
        union |= mask
    return union


def _write_label_masks(
    per_annotator_masks: list[list[np.ndarray]],
    size: int,
    *,
    semantic_path: Path,
    instance_path: Path | None,
    annotator_path: Path | None,
    probability_path: Path | None,
) -> None:
    """Write one frame's label masks, in several-labeller or single-labeller mode.

    ``per_annotator_masks`` is one list of per-contrail boolean masks per labeller. In
    several-labeller mode (``annotator_path`` set) it writes one union mask per labeller
    (``annotator_masks`` uint8 ``(H, W, A)``), their mean (``probability_masks`` float32
    ``(H, W)``) and the majority vote (``semantic_masks`` PNG, ``(votes * 2 > A) * 255``). In
    single-labeller mode it writes the union as the semantic PNG and, when ``instance_path`` is
    set, each kept contrail as one instance slice.
    """
    if annotator_path is not None and probability_path is not None:
        labellers = len(per_annotator_masks)
        unions = [_union_of(masks, size) for masks in per_annotator_masks]
        stack = np.stack(unions, axis=-1)  # (H, W, A) bool, one union mask per labeller
        _save_npy(annotator_path, stack.astype(np.uint8))
        _save_npy(probability_path, stack.mean(axis=-1).astype(np.float32))
        semantic = (stack.sum(axis=-1) * 2 > labellers) * 255
        _save_image(semantic_path, Image.fromarray(semantic.astype(np.uint8), mode="L"))
        return
    masks = per_annotator_masks[0]
    union = _union_of(masks, size)
    _save_image(semantic_path, Image.fromarray((union * 255).astype(np.uint8), mode="L"))
    if instance_path is not None:
        _save_npy(instance_path, _instance_array(masks, size))


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


def _prepare_sequences(
    sequences: list[Sequence], instances: bool
) -> tuple[bool, list[Sequence], dict[str, tuple[str, ...]], tuple[str, ...] | None]:
    """Detect several-labeller mode, validate it and merge labellers into one sequence each.

    Returns ``(multi, effective_sequences, annotators_by_name, global_annotators)``. In
    single-labeller mode ``multi`` is ``False`` and the sequences pass through unchanged.

    Raises
    ------
    ValueError
        If several labellers are found but ``instances`` is requested, the sequences do not all
        share the same set of labellers, or one sequence's labellers disagree on the frame list.
    """
    by_name: dict[str, list[Sequence]] = defaultdict(list)
    for seq in sequences:
        by_name[seq.name].append(seq)
    is_multi = any(
        len({seq.annotator for seq in labs if seq.annotator}) > 1 for labs in by_name.values()
    )
    if not is_multi:
        return False, sequences, {}, None
    if instances:
        msg = "instance masks with several labellers are not supported yet"
        raise ValueError(msg)
    annotator_sets = {
        name: tuple(sorted({seq.annotator for seq in labs if seq.annotator}))
        for name, labs in by_name.items()
    }
    distinct = set(annotator_sets.values())
    if len(distinct) > 1:
        msg = f"sequences have differing annotator sets {sorted(distinct)}"
        raise ValueError(msg)
    annotator_order = next(iter(distinct))
    merged: list[Sequence] = []
    annotators_by_name: dict[str, tuple[str, ...]] = {}
    for name in sorted(by_name):
        labs = by_name[name]
        titles = [frame.title for frame in labs[0].frames]
        for lab in labs[1:]:
            if [frame.title for frame in lab.frames] != titles:
                msg = f"sequence {name!r}: labellers have differing frame title lists"
                raise ValueError(msg)
        merged.append(_merge_sequence(name, labs, annotator_order))
        annotators_by_name[name] = annotator_order
    return True, merged, annotators_by_name, annotator_order


def _merge_sequence(name: str, labs: list[Sequence], annotator_order: tuple[str, ...]) -> Sequence:
    """Merge one sequence's per-labeller exports into a single multi-annotator Sequence.

    The merged sequence keeps the base ``project`` (labeller suffix stripped) and frame
    metadata from the first labeller; each frame's ``annotators`` holds every labeller's
    contrails in the sorted labeller order.
    """
    base = labs[0]
    base_frames_of = {frame.title: frame for frame in base.frames}
    frames_by_annotator = {
        seq.annotator: {frame.title: frame for frame in seq.frames} for seq in labs
    }
    frames = []
    for title in [frame.title for frame in base.frames]:
        per_annotator = tuple(
            frames_by_annotator[annot][title].contrails for annot in annotator_order
        )
        base_frame = base_frames_of[title]
        frames.append(
            LabelledFrame(
                title=title,
                npy=base_frame.npy,
                labelled_size=base_frame.labelled_size,
                contrails=(),
                annotators=per_annotator,
            )
        )
    suffix = annotator_of(base.project)
    base_project = base.project[: -len(suffix)] if suffix else base.project
    return Sequence(
        name=name,
        project=base_project,
        utm_zone=base.utm_zone,
        frame_band=base.frame_band,
        start=base.start,
        end=base.end,
        frames=tuple(frames),
        annotator=None,
    )


def _load_split(split_from: Path, groups: dict[str, str]) -> dict[str, str]:
    """Use the group-id -> split assignment recorded in an existing ``metadata/split.json``.

    ``groups`` is the ``day_groups`` output (sequence name -> group id); the split.json ``groups``
    mapping is keyed by group id, which is what this returns.

    Raises
    ------
    ValueError
        If the file is missing, is not JSON, has no ``groups`` mapping, or does not record every
        group id computed for the current labels.
    """
    path = split_from / "metadata" / "split.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        msg = f"split_from has no metadata/split.json: {path}"
        raise ValueError(msg) from None
    except json.JSONDecodeError as exc:
        msg = f"split_from metadata/split.json is not valid JSON: {path} ({exc})"
        raise ValueError(msg) from exc
    loaded = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(loaded, dict):
        msg = f"split_from metadata/split.json has no 'groups' mapping: {path}"
        raise ValueError(msg)
    group_ids = set(groups.values())
    missing = [gid for gid in group_ids if gid not in loaded]
    if missing:
        example = ", ".join(sorted(missing)[:3])
        msg = f"split_from is missing {len(missing)} of {len(group_ids)} group(s), e.g. {example}"
        raise ValueError(msg)
    return {gid: str(loaded[gid]) for gid in sorted(group_ids)}


def _build_plan(
    sequences: list[Sequence],
    groups: dict[str, str],
    assignment: dict[str, str],
    images_dir: Path,
    output_dir: Path,
    size: int,
    overwrite: bool,
    instances: bool,
    link_bands_from: Path | None,
) -> list[_PlanItem]:
    """Build one :class:`_PlanItem` per frame with an image, resolved to its split and job.

    A frame with ``annotators`` set is a several-labeller frame (one union mask per labeller);
    otherwise it is single-labeller (its ``contrails`` become the one annotator).
    """
    plan: list[_PlanItem] = []
    for sequence in sequences:
        group_id = groups[sequence.name]
        split = assignment[group_id]
        disk_split = "train" if split == "val" else split
        for frame in sequence.frames:
            if frame.npy is None:
                continue
            bands_path = output_dir / "spectral_bands" / disk_split / f"{frame.title}.npy"
            semantic_path = output_dir / "semantic_masks" / disk_split / f"{frame.title}.png"
            instance_path = annotator_path = probability_path = bands_source = None
            if frame.annotators is not None:
                annotator_path = output_dir / "annotator_masks" / disk_split / f"{frame.title}.npy"
                probability_path = (
                    output_dir / "probability_masks" / disk_split / f"{frame.title}.npy"
                )
                annotators_contrails = tuple(
                    tuple(
                        (c.key, c.confidence, tuple(tuple(point) for point in c.polygons))
                        for c in annotator
                    )
                    for annotator in frame.annotators
                )
            else:
                single: _AnnotatorContrails = tuple(
                    (c.key, c.confidence, tuple(tuple(point) for point in c.polygons))
                    for c in frame.contrails
                )
                annotators_contrails = (single,)
                if instances:
                    instance_path = (
                        output_dir / "instance_masks" / disk_split / f"{frame.title}.npy"
                    )
            if link_bands_from is not None:
                bands_source = str(
                    link_bands_from / "spectral_bands" / disk_split / f"{frame.title}.npy"
                )
            job = _FrameJob(
                id=frame.title,
                image_path=str(images_dir / frame.npy),
                bands_path=str(bands_path),
                semantic_path=str(semantic_path),
                instance_path=str(instance_path) if instance_path else None,
                annotator_path=str(annotator_path) if annotator_path else None,
                probability_path=str(probability_path) if probability_path else None,
                bands_source=bands_source,
                annotators_contrails=annotators_contrails,
                size=size,
                overwrite=overwrite,
            )
            plan.append(_PlanItem(sequence, frame, split, disk_split, job))
    return plan


def convert(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    size: int = 1024,
    workers: int = 8,
    seed: int = 0,
    overwrite: bool = False,
    link_bands_from: Path | None = None,
    split_from: Path | None = None,
    instances: bool = True,
) -> dict[str, Any]:
    """Convert the label exports and images under ``labels_dir`` / ``images_dir`` to ``output_dir``.

    Parameters
    ----------
    labels_dir
        Directory of Encord label ``*.json`` exports, one per (sequence, labeller).
    images_dir
        Directory holding the per-frame ``.npy`` images referenced by the exports (unused when
        ``link_bands_from`` hard-links the bands instead).
    output_dir
        Dataset root to write in the DeepContrail format v1 layout.
    size
        Output side length; the image is block-meaned to it when larger (must divide evenly).
    workers
        Worker processes for the per-frame conversion; run inline when ``1``.
    seed
        Seed for the day-group split search (ignored when ``split_from`` is given).
    overwrite
        Rewrite a frame even when its files already exist.
    link_bands_from
        When set, hard-link ``<root>/spectral_bands/<split>/<id>.npy`` from an existing dataset
        instead of computing the bands from the images; that dataset's ``bands.json`` is copied.
    split_from
        When set, reuse the ``groups`` of an existing ``<root>/metadata/split.json`` instead of
        searching a new split (the report is still computed and written).
    instances
        Write ``instance_masks`` in single-labeller mode; must be ``False`` with several labellers.

    Returns
    -------
    dict
        A summary with frames written and skipped, frames without an image, the kept contrail
        count, dropped empty contrails, whether several labellers were combined and the
        per-split frame fractions.

    Raises
    ------
    ValueError
        If two frames share a scene id, a group mixes projects, several labellers are asked to
        keep instance masks or disagree on their set or frame list, ``size`` is larger than an
        image side, an image side is not divisible by ``size``, ``split_from`` is missing a
        group, or a linked band file is missing.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    link_bands_from = None if link_bands_from is None else Path(link_bands_from)
    split_from = None if split_from is None else Path(split_from)

    raw_sequences = [read_sequence(path) for path in sorted(labels_dir.glob("*.json"))]
    multi, sequences, annotators_by_name, global_annotators = _prepare_sequences(
        raw_sequences, instances
    )

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

    if split_from is not None:
        assignment = _load_split(split_from, groups)
    else:
        assignment = make_split(
            sequences,
            groups,
            test_fraction=DEFAULT_TEST_FRACTION,
            val_fraction=DEFAULT_VAL_FRACTION,
            seed=seed,
        )

    bands_json_source = None
    if link_bands_from is not None:
        bands_json_source = link_bands_from / "spectral_bands" / "bands.json"
        if not bands_json_source.is_file():
            msg = f"the linked dataset has no spectral_bands/bands.json: {bands_json_source}"
            raise ValueError(msg)

    if multi:
        labels = ("semantic", "probability", "annotator")
        label_subdirs = (
            "spectral_bands",
            "semantic_masks",
            "annotator_masks",
            "probability_masks",
        )
    elif instances:
        labels = ("semantic", "instance")
        label_subdirs = ("spectral_bands", "semantic_masks", "instance_masks")
    else:
        labels = ("semantic",)
        label_subdirs = ("spectral_bands", "semantic_masks")

    plan = _build_plan(
        sequences,
        groups,
        assignment,
        images_dir,
        output_dir,
        size,
        overwrite,
        instances,
        link_bands_from,
    )

    for sub in label_subdirs:
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
        kept_by_annotator = result["kept_by_annotator"]
        kept_contrails += sum(len(kept) for kept in kept_by_annotator)
        if item.frame.annotators is not None:
            label_contrails += sum(len(annotator) for annotator in item.frame.annotators)
        else:
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
        scene: dict[str, Any] = {
            "split": item.split,
            "project": item.sequence.project,
            "sequence": item.sequence.name,
            "group": groups[item.sequence.name],
            "time": _time_from_title(item.frame.title),
            "utm_zone": item.sequence.utm_zone,
            "frame_band": item.sequence.frame_band,
            "labelled_size": item.frame.labelled_size,
            "nan_fraction": result["nan_fraction"],
        }
        if multi:
            annotators = annotators_by_name[item.sequence.name]
            scene["contrails_by_annotator"] = {
                annot: kept_by_annotator[index] for index, annot in enumerate(annotators)
            }
            scene["annotators"] = list(annotators)
        else:
            scene["contrails"] = kept_by_annotator[0]
        scenes[item.frame.title] = scene

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
        labels=labels,
        annotators=global_annotators,
        multi=multi,
        bands_json_source=bands_json_source,
    )

    total = len(results)
    summary: dict[str, Any] = {
        "frames_written": frames_written,
        "frames_skipped": frames_skipped,
        "frames_without_image": n_no_image,
        "contrails": kept_contrails,
        "dropped_empty_contrails": label_contrails - kept_contrails,
        "several_labellers": multi,
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
    *,
    labels: tuple[str, ...],
    annotators: tuple[str, ...] | None,
    multi: bool,
    bands_json_source: Path | None,
) -> None:
    """Write bands.json, dataset_info.json, scenes.json, split.json and the split lists."""
    n_train_dir = len(train_ids) + len(val_ids)
    n_test = len(test_ids)

    bands_manifest = output_dir / "spectral_bands" / "bands.json"
    if bands_json_source is not None:
        bands_manifest.write_text(
            Path(bands_json_source).read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        bands_manifest.write_text(json.dumps(band_manifest(), indent=2) + "\n", encoding="utf-8")

    info: dict[str, Any] = {
        "format_version": 1,
        "source": "MTG-I1 FCI contrail labels (EUROCONTROL, labelled in Encord)",
        "records": {"train": n_train_dir, "test": n_test},
        "labels": list(labels),
        "recommended_pixel_shift": 0.0,
        "temporal": False,
        "naming": "image_title",
    }
    if multi and annotators is not None:
        # Declare "annotators" in dataset_info only if DatasetInfo accepts it as a known field;
        # it does not (a reader would warn on the unknown key), so the list stays in scenes.json.
        probe = DatasetInfo.model_validate({**info, "annotators": list(annotators)})
        if "annotators" not in (probe.model_extra or {}):
            info["annotators"] = list(annotators)
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
    parser.add_argument(
        "--link-bands-from",
        type=Path,
        default=None,
        help="Hard-link bands from this converted dataset's spectral_bands/<split>/<id>.npy "
        "instead of computing them; its bands.json is copied (pair with --split-from).",
    )
    parser.add_argument(
        "--split-from",
        type=Path,
        default=None,
        help="Reuse this dataset's metadata/split.json group assignment instead of searching one.",
    )
    parser.add_argument(
        "--no-instances",
        dest="instances",
        action="store_false",
        default=True,
        help="Do not write instance_masks (implied with several labellers).",
    )
    return parser.parse_args(argv)


def _print_summary(summary: dict[str, Any]) -> None:
    frac = summary["split_fractions"]
    labellers = "several labellers" if summary["several_labellers"] else "one labeller"
    print(
        f"converted {summary['frames_written']} frames, skipped {summary['frames_skipped']} "
        f"({labellers}; no image: {summary['frames_without_image']}); {summary['contrails']} "
        f"contrails ({summary['dropped_empty_contrails']} dropped as empty at the output size)"
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
        link_bands_from=args.link_bands_from,
        split_from=args.split_from,
        instances=args.instances,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
