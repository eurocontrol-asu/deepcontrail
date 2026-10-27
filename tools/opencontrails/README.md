# deepcontrail-opencontrails

Converts the raw OpenContrails GOES-16 TFRecords into the on-disk layout DeepContrail reads.
It is one producer of that layout; it is not part of the library and needs TensorFlow.

```bash
uv sync --group convert
uv run deepcontrail-convert-opencontrails --tfrecord-dir /path/to/20250305/tfrecords --output-dir /path/to/dataset
```
