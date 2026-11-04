# deepcontrail-opencontrails

Converts the raw [OpenContrails](https://sites.research.google/gr/contrails/public-datasets/) GOES-16
TFRecords (bucket [`goes_contrails_dataset`](https://storage.cloud.google.com/goes_contrails_dataset))
into the [DeepContrail dataset format](../../packages/core). It is one producer of
that format, not part of the library, and it needs TensorFlow.

```bash
uv sync --group convert
uv run deepcontrail-convert-opencontrails \
    --tfrecord-dir /path/to/20250305/tfrecords --output-dir /path/to/dataset
uv run deepcontrail-validate /path/to/dataset --deep
```

- Training shards go to `train/`, validation shards to `test/` (the holdout).
- Each record is named by its OpenContrails `record_id`, so a subset of shards (`--file-glob`)
  converts to the same names as a full run.
- Conversion resumes: complete records are skipped (`--no-skip-existing` rewrites them).
- `spectral_bands/<id>.npy` holds 10 channels: 9 infrared ABI bands and their split-window
  difference. `bands.json` maps each key to its ABI band (`data_10um` is band 13, `cloud_top`
  band 11). The preview JPEGs use the standard ash RGB.
- `metadata/dataset_info.json` records the export and its pixel shift: `20250305` masks are
  aligned with the images (0.0), `20230419` masks follow the Kaggle convention (0.5).
- `--metadata-only` rewrites just the two metadata files of an already converted dataset,
  without the TFRecords or TensorFlow.
