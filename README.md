# DeepContrail

Open-source library for **contrail detection, segmentation, tracking, and flight attribution** in satellite imagery.

> **Status:** early development — package skeleton, tooling, and test suite are in place; implementation is being built out incrementally.

## Goal

Provide a clean, well-typed, and reusable library for detecting contrails in satellite imagery. Development follows a phased roadmap:

1. **Semantic segmentation** — detect contrail pixels in GOES-16 imagery (active)
2. **Instance segmentation** — separate individual contrails (MTG)
3. **Tracking** — follow contrails across time steps
4. **Attribution** — assign ADS-B flight trajectories to detected contrails
5. **Altitude estimation** — infer contrail altitude from imagery

## Installation

The repository is a [uv](https://github.com/astral-sh/uv) workspace of separately
installable packages. From a clone:

```bash
uv sync                    # core + segmentation, dev tools, training scripts' dependencies
uv sync --group convert    # + the OpenContrails converter (TensorFlow)
```

| Package | Import | Contents |
|---|---|---|
| `deepcontrail-core` | `deepcontrail.core` | Shared and torch-free: dataset config, spectral bands, composites, splits, Dice/IoU |
| `deepcontrail-segmentation` | `deepcontrail.segmentation` | Semantic segmentation: dataset, transforms, models, losses, training, TTA, evaluation |
| `deepcontrail-opencontrails` | `deepcontrail.opencontrails` | Converter from the OpenContrails TFRecords (tool, not part of the library) |

## Layout

```
packages/
├── core/            # deepcontrail-core:          src/deepcontrail/core/, tests/
└── segmentation/    # deepcontrail-segmentation:  src/deepcontrail/segmentation/, tests/,
                     #   scripts/ (train, evaluate; not shipped), configs/experiment.yaml
tools/
└── opencontrails/   # deepcontrail-opencontrails: src/deepcontrail/opencontrails/, tests/
tests/               # workspace-level checks (versions, licences, namespace layout)
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All four must be green before opening a pull request.

## License

Licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE).
