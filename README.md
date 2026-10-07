# DeepContrail

Open-source library for the **semantic segmentation of contrails** (vapor trails left by
aircraft) from satellite imagery.

> **Status:** early development. The package skeleton, tooling, and testing setup are in
> place; the implementation is being built out incrementally. See the working rules in
> [`CLAUDE.md`](CLAUDE.md) and the roadmap in the project plan.

## Goal

Provide a clean, well-typed, and reusable library for detecting contrails in satellite
imagery. The first milestone is a semantic-segmentation model on the GOES-16 dataset,
measured with the Dice coefficient.

## Installation

DeepContrail is managed with [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv
uv pip install -e ".[dev]"
```

`[dev]` pulls the development tooling (pytest, ruff, pyright). The training stack
(torch, segmentation_models_pytorch, …) is installed as it is needed.

## Layout

```
src/deepcontrail/      # the shipped package (skeleton)
├── __init__.py        # version + public exports
├── py.typed           # PEP 561 typing marker
├── data/              # dataset access (Phase 1: GOES-16)
├── models/            # model construction (Phase 1: U-Net family)
└── metrics/           # evaluation (Phase 1: Dice)
```

Development happens in the open; implementation code is added milestone by milestone.

## Development

This project enforces a consistent code style and quality bar. Before opening a pull
request, make sure:

- `ruff check .` is clean — `ruff format` has already been applied;
- `pyright` passes on `src/`;
- `pytest` passes.

The full working rules (coding structure, TDD, ML quality, git workflow) live in
[`CLAUDE.md`](CLAUDE.md).

## License

This project is licensed under the European Union Public Licence v1.2 (EUPL-1.2).
See [`LICENSE`](LICENSE).
