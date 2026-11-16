"""Structured view of one MTG/FCI sequence's Encord label export.

An Encord export is one JSON file per labelled sequence (48 frames, 10 minutes apart). Human
labellers drew polygons on the frames; a single contrail may be split across several fragment
polygons and those fragments are ONE instance. This module reads that export into a few small,
dependency-free dataclasses the converter builds on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

CONFIDENCE_ORDER = ("Low", "Medium", "High")
"""Track confidence levels, from least to most confident."""

_MTG_NAME = re.compile(r"^MTG_(\d{12})_(\d{12})_zone(\d+)_frame(\d+)$")
_MTI1_NAME = re.compile(r"^MTI1_(\d{8})_(\d{4})_zone(\d+)_frame(\d+)$")


@dataclass(frozen=True)
class Contrail:
    """One labelled contrail instance: all its fragment polygons under a single key.

    Parameters
    ----------
    key
        The contrail key in a frame's ``contrails`` dict.
    polygons
        Fragment polygons, each a tuple of normalised ``(x, y)`` points (``x`` = column,
        ``y`` = row, both in ``0..1`` from the top-left image corner).
    confidence
        Highest track confidence of this contrail (``"High" > "Medium" > "Low"``), or
        ``None`` when no track carries a known level.
    """

    key: str
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    confidence: str | None


@dataclass(frozen=True)
class LabelledFrame:
    """One frame of a sequence, with the polygons drawn on it.

    Parameters
    ----------
    title
        ``image_title`` without its extension — the scene id.
    npy
        Path to the frame image (``image_npy``), or ``None`` when the frame has no image.
    labelled_size
        The image width the labeller drew on (512 or 1024).
    contrails
        Contrails on this frame, sorted by key.
    """

    title: str
    npy: str | None
    labelled_size: int
    contrails: tuple[Contrail, ...]


@dataclass(frozen=True)
class Sequence:
    """One labelled MTG/FCI sequence — a single Encord export."""

    name: str
    project: str
    utm_zone: int
    frame_band: int
    start: datetime
    end: datetime
    frames: tuple[LabelledFrame, ...]


def parse_sequence_name(name: str) -> tuple[datetime, datetime | None, int, int]:
    """Parse a sequence ``name`` into ``(start, end, utm_zone, frame_band)``.

    Two styles are recognised:

    - ``MTG_<YYYYmmddHHMM>_<YYYYmmddHHMM>_zone<z>_frame<b>`` — start and end are both present.
    - ``MTI1_<YYYYmmdd>_<HHMM>_zone<z>_frame<b>`` — only a start; ``end`` is ``None`` and the
      caller defaults it to ``start + 8 h``.

    Times are UTC-naive.

    Raises
    ------
    ValueError
        If ``name`` matches neither style.
    """
    match = _MTG_NAME.match(name)
    if match:
        start = datetime.strptime(match.group(1), "%Y%m%d%H%M")
        end = datetime.strptime(match.group(2), "%Y%m%d%H%M")
        return start, end, int(match.group(3)), int(match.group(4))
    match = _MTI1_NAME.match(name)
    if match:
        start = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")
        return start, None, int(match.group(3)), int(match.group(4))
    msg = f"unrecognised sequence name: {name!r}"
    raise ValueError(msg)


def contrail_confidence(key: str, label: dict) -> str | None:
    """Return the highest confidence over the tracks grouped under ``contrail_groups[key]``.

    ``label`` is the whole export: ``contrail_groups[key]`` lists track ids and each track's
    level is ``tracks[tid]["static"]["Confidence Level"]``.

    Returns
    -------
    str | None
        The highest of ``"Low" < "Medium" < "High"`` found, or ``None`` when no track carries
        a known level (missing ``static``, missing key, or an unknown value).
    """
    groups = label.get("contrail_groups") or {}
    tracks = label.get("tracks") or {}
    levels = []
    for track_id in groups.get(key, []):
        track = tracks.get(track_id)
        if isinstance(track, dict) and isinstance(track.get("static"), dict):
            level = track["static"].get("Confidence Level")
            if level in CONFIDENCE_ORDER:
                levels.append(level)
    if not levels:
        return None
    return max(levels, key=CONFIDENCE_ORDER.index)


def read_sequence(path: str | Path) -> Sequence:
    """Read one Encord label export into a :class:`Sequence`.

    Polygons with fewer than three points are dropped; a contrail left without polygons is
    dropped. ``end`` is the parsed end time, or ``start + 8 h`` when the name has none. The
    sequence's ``utm_zone`` and ``frame_band`` come from the ``sequence`` name.

    Raises
    ------
    ValueError
        If the ``sequence`` name is not recognised (see :func:`parse_sequence_name`).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = data["sequence"]
    start, end, utm_zone, frame_band = parse_sequence_name(name)
    if end is None:
        end = start + timedelta(hours=8)

    frames = tuple(
        LabelledFrame(
            title=_without_extension(str(image["image_title"])),
            npy=image.get("image_npy"),
            labelled_size=int(image["width"]),
            contrails=_contrails_of(image.get("contrails") or {}, data),
        )
        for image in data.get("images", [])
    )

    return Sequence(
        name=name,
        project=str(data.get("project_name", "")),
        utm_zone=utm_zone,
        frame_band=frame_band,
        start=start,
        end=end,
        frames=frames,
    )


def _without_extension(title: str) -> str:
    """Drop the final ``.ext`` of an ``image_title`` (``a.b.png`` -> ``a.b``)."""
    return title.rsplit(".", 1)[0] if "." in title else title


def _contrails_of(contrails: dict, label: dict) -> tuple[Contrail, ...]:
    """Merge the fragment polygons under each key into one :class:`Contrail`, sorted by key."""
    out = []
    for key in sorted(contrails):
        polygons = []
        for polygon in contrails[key]:
            points = tuple((float(x), float(y)) for x, y in polygon)
            if len(points) >= 3:
                polygons.append(points)
        if polygons:
            out.append(
                Contrail(
                    key=key,
                    polygons=tuple(polygons),
                    confidence=contrail_confidence(key, label),
                )
            )
    return tuple(out)
