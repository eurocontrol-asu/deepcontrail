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

```bash
# Inference only
pip install deepcontrail

# Full training stack
pip install "deepcontrail[train]"

# Development
pip install "deepcontrail[dev]"
```

With [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv
uv pip install -e ".[dev]"
```

## Layout

```
src/deepcontrail/   # shipped library
├── data/           # dataset loaders (GOES-16, later MTG)
├── models/         # model construction
├── losses/         # training losses
├── metrics/        # evaluation (Dice, …)
└── training/       # training infrastructure
scripts/            # training and evaluation scripts (not shipped)
configs/            # experiment YAMLs
tests/              # unit and integration tests
```

## Development

```bash
ruff check .          # lint
ruff format --check . # format
pyright               # types
uv run pytest         # tests
```

All three must be green before opening a pull request.

## License

Licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE).
