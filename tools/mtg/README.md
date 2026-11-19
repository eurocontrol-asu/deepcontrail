# deepcontrail-mtg

Converts **MTG-I1 FCI** contrail labels from Encord exports (one JSON file per labelled
sequence of 48 frames, 10 minutes apart) and the matching per-frame `.npy` images into the
[DeepContrail dataset format v1][core]. It is one producer of that format, not part of it.

[core]: https://github.com/eurocontrol-asu/deepcontrail/tree/main/packages/core

## Input

- **Labels.** One Encord export `*.json` per sequence. `read_sequence()` parses the sequence
  name (`MTG_<start>_<end>_zone<z>_frame<b>`), the per-frame `image_title` / `image_npy` /
  `width`, and merges the fragment polygons of each `contrail_groups` key into one instance.
- **Images.** The per-frame `.npy` files the exports point at, each a float32 `(S, S, 3)`
  toolkit composite in the order `[T12.3 - T10.5, T10.5 - T8.7, T10.5]`, `NaN` = no data.

## Command

```bash
deepcontrail-convert-mtg --labels <label_dir> --images <image_dir> --out <dataset_dir> \
    [--size 1024] [--workers 8] [--seed 0] [--overwrite] \
    [--link-bands-from <dataset_dir>] [--split-from <dataset_dir>] [--no-instances] \
    [--neighbours-only]
```

`--size` block-means larger images down to that side (the side must divide evenly). Frames are
converted in parallel (`--workers`); a run resumes by skipping frames whose files already exist
(`--overwrite` rewrites them). `--link-bands-from` and `--split-from` reuse an already converted
dataset (see below). `--neighbours-only` skips conversion and only (re)builds
`metadata/neighbours.json` from `<dataset_dir>`'s `scenes.json` (see below).

## What is written

```
<out>/
├── spectral_bands/{train,test}/{id}.npy   float32 (H, W, 3) bands, NaN = no data
├── spectral_bands/bands.json              channel manifest (format v1)
├── semantic_masks/{train,test}/{id}.png   uint8 union of the instances, 0 / 255 (mode "L")
├── instance_masks/{train,test}/{id}.npy   int8 (N, H, W, 1), 1 inside contrail n, 0 outside
├── metadata/dataset_info.json             source, record counts, labels, pixel shift
├── metadata/scenes.json                   per-scene provenance and kept contrails
├── metadata/neighbours.json               previous/next frame of each scene (temporal)
├── metadata/split.json                    the seeded split, its groups and a report
├── splits_train.txt / splits_test.txt     the ids written to each on-disk split
└── splits_val.txt                         the val ids (they live under train/ on disk)
```

The three channels are stored in `BAND_KEYS` order — `ir_105` (T10.5), `ir_87` (T8.7),
`ir_123_minus_ir_105` (T12.3 − T10.5) — and `bands.json` declares `"nodata": "nan"`, so `NaN`
marks pixels outside the view. Contrails whose mask is empty at the output size are dropped.

The standard ash false-colour recipe (`bands.FALSE_COLOR`) is ready to drop into a data config:

```yaml
false_color:
  channels:
    - {bands: [ir_123_minus_ir_105], bounds: [-4, 2]}
    - {bands: [ir_105, ir_87],        bounds: [-4, 5]}
    - {bands: [ir_105],               bounds: [243, 303]}
```

`metadata/dataset_info.json` declares both labels (`semantic`, `instance`),
`recommended_pixel_shift: 0.0` and `temporal: false`.

`metadata/neighbours.json` links each scene to the previous and next frame of its sequence, in
time order (frames without an image are skipped). A sequence never spans two splits, so the two
neighbours of a scene are always in the same on-disk split directory. A downstream data config
can then set `context_frames: k` to stack the `2k+1` frames around each scene into
`3*(2k+1)` channels; a scene at a sequence end repeats its nearest frame. Rebuild the file for an
already converted root with `deepcontrail-convert-mtg --out <dataset_dir> --neighbours-only`.

## The split

Frames 10 minutes apart are near-duplicates, so the split holds out **whole calendar days**:

- A *group* is a calendar day, merged (union-find) with every sequence that overlaps another in
  space and time — the same `frame_band`, an `utm_zone` difference of at most one, and
  overlapping time windows (even across midnight). A merged group that mixes projects is an
  error.
- Groups are assigned to `train` / `val` / `test` by a **seeded random search** that keeps the
  assignment best matching the targets — about **15 % test** and **10 % val** — weighted by
  frames and by the data's contrast (contrails per frame, share of low-confidence contrails).
  Every project keeps at least one whole day in test. The search is deterministic for a seed.

`val` scenes are written under `train/` on disk; use **`splits_val.txt`** as the data config's
`val_ids` — a whole-day list, so a frame-level split cannot leak near-duplicates.

## Several labellers

The label folder can hold **one JSON per (sequence, labeller)**: the same `sequence` name once
per person, each `project_name` ending in a different `_L<k>` (e.g. `MTG_S10_L1`,
`MTG_S10_L3`) and the same frame list. Contrail ids are **not** comparable across people, so the
labellers are combined at the **pixel** level — each frame gets, per labeller, the union of that
labeller's polygons:

- `annotator_masks/{split}/{id}.npy` — uint8 `(H, W, A)`, one binary union mask per labeller, in
  the sorted labeller order (`L1`, `L2`, …).
- `probability_masks/{split}/{id}.npy` — float32 `(H, W)`, the mean over labellers (their
  agreement).
- `semantic_masks/{split}/{id}.png` — the **majority**: a pixel is drawn when more than half
  vote for it (`votes * 2 > A`), values `0 / 255`.

`dataset_info.json` then declares `["semantic", "probability", "annotator"]` (no `instance`),
so instance masks are **not** written — pass `--no-instances` (or they are implied); asking for
instance masks with several labellers is an error. `scenes.json` records each labeller's kept
contrails under `contrails_by_annotator` and the labeller order under `annotators`. All labellers
of a sequence must share the same set of `_L<k>` suffixes and the same frame list, or the
conversion fails. The split statistics average the per-frame contrail and low-confidence counts
over labellers.

## Reusing an already converted dataset

Two options turn an existing dataset into the input of a new run instead of recomputing:

- `--link-bands-from <dataset_dir>` hard-links `spectral_bands/<split>/<id>.npy` from that
  dataset (rather than recomputing the bands from the images) and copies its `bands.json`. The
  id must sit in the same `<split>` directory (`train`/`test`) as in the source.
- `--split-from <dataset_dir>` reuses that dataset's `metadata/split.json` group assignment
  instead of searching a new one (the report is still computed and written). Pair it with
  `--link-bands-from` to reproduce a dataset's bands and split exactly.

## Not published on PyPI yet

Install it from the source checkout (it depends on `deepcontrail-core` from this workspace);
the command above is `deepcontrail.mtg.convert:main`.
