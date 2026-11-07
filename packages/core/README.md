# deepcontrail-core

Torch-free foundations of [DeepContrail](https://github.com/eurocontrol-asu/deepcontrail): the
**dataset format** every task reads, its validation, false-colour composites built from named
bands, the train/validation/fold partition, and Dice/IoU metrics.

Depends only on numpy, Pillow, pydantic and PyYAML. `import deepcontrail.core` never imports torch.

## The DeepContrail dataset format (version 1)

The library reads this layout and nothing else. Any producer can write it; for OpenContrails
the [converter](../../tools/opencontrails) does.

```
<root>/
├── metadata/dataset_info.json               required: source, record counts, labels, pixel shift
├── spectral_bands/bands.json                required: what each channel of the arrays holds
├── spectral_bands/{train,test}/<id>.npy     required: (H, W, C) float32, e.g. brightness temperatures (K)
├── semantic_masks/{train,test}/<id>.png     binary consensus mask: 0 background, 255 contrail (1 accepted)
├── probability_masks/{train,test}/<id>.npy  (H, W) float32 annotator agreement in [0, 1]
├── annotator_masks/{train,test}/<id>.npy    (H, W, A) uint8 {0, 1}, one mask per annotator
├── instance_masks/{train,test}/<id>.npy     (N, H, W, A) int8: 1 = pixel of instance n for annotator a,
│                                             0 = not, -1 = instance not drawn by that annotator
├── images_semantic/{train,test}/<id>.jpg    optional preview for people; never read by the library
└── splits_{train,test}.txt                  optional: the labelled ids of each split
```

- **Splits.** `test/` is a holdout. The validation set is drawn from `train/` at load time:
  the sorted ids are shuffled with the seed and the first `val_fraction` go to validation (or
  k-fold with `fold` / `num_folds`). Training runs record the exact lists and their fingerprints.
- **Ids** are unique across splits. Unlabelled neighbouring time steps of a scene may be stored
  as `<id>_b<k>.npy` / `<id>_a<k>.npy` (k steps before / after); they need no labels. Labelled
  scene ids must not end in `_a<k>` or `_b<k>` (reserved for context frames).
- **Labels** are only required for the task at hand: `semantic` for segmentation, `instance`
  for instance segmentation. Every label kind declared in `dataset_info.json` must cover every
  labelled scene, at the size of its spectral array.

All JSON and text files (`bands.json`, `dataset_info.json`, `splits_*.txt`) must be UTF-8 encoded.

### `spectral_bands/bands.json`

```json
{
  "format_version": 1,
  "sensor": "GOES-16 ABI",
  "dtype": "float32",
  "channels": [
    {"index": 0, "key": "data_10um", "units": "K", "sensor_band": "13", "wavelength_um": 10.33},
    {"index": 1, "key": "data_11um", "units": "K", "sensor_band": "14", "wavelength_um": 11.19}
  ]
}
```

`key` (unique, snake_case) is the name configs use; `index` is the channel's position. The
optional `sensor_band`, `wavelength_um`, `name` and `derived_from` (`[a, b]` for a channel
stored as `a - b`) describe the band for people; the library never interprets them. Another
sensor declares its own keys (for example `ir_105`, `ir_123` for MTG FCI).

### `metadata/dataset_info.json`

```json
{
  "format_version": 1,
  "source": "OpenContrails GOES-16 ABI TFRecords",
  "records": {"train": 20088, "test": 1879},
  "labels": ["semantic", "probability", "annotator", "instance"],
  "recommended_pixel_shift": 0.0
}
```

`recommended_pixel_shift` is the shift (pixels, right and down) that aligns the images with the
masks, or `null` if unknown. The data config's `pixel_shift` should use it; a dataset warns when
they differ.

Read both with `read_band_manifest(root)` and `read_dataset_info(root)`; a missing or invalid file
raises `DatasetContractError` naming the file.

Instance masks are read with `deepcontrail.core.readers.load_instance_masks(path)`, which raises
`ValueError` on an array that is not in the format above; `extract_all_annotator_instances(raw)`
turns one into each annotator's list of boolean contrail masks.

## Composites

A composite is three channels built from named bands, each one band or the difference of two,
scaled from `bounds` to [0, 1]. The recipe lives in the experiment config:

```yaml
data:
  false_color:
    channels:
      - {bands: [data_12um, data_11um], bounds: [-4, 2]}    # R
      - {bands: [data_11um, cloud_top], bounds: [-4, 5]}    # G
      - {bands: [data_11um],            bounds: [243, 303]} # B
```

`false_color_from_spectral(bands, recipe, manifest)` builds it; a band key the dataset does not
declare is an error listing the available keys.

## Validation

```bash
deepcontrail-validate <root> [--task segmentation|instance] [--deep]
```

The default checks read only the metadata and the file names (seconds): both metadata files,
the task's labels, every labelled scene has its labels and no label lacks its scene, record
counts and split lists. `--deep` also reads every file: arrays load with the declared channels
and finite values, masks match their scene's size and hold valid values, all scenes share one
size. It exits non-zero on any issue. From Python: `validate_dataset(root, task=..., deep=...)`.
