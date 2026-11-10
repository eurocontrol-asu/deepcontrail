# Contributing to DeepContrail

Thank you for helping. Bug reports, questions, documentation fixes and code are all welcome.

## Before you start

- **Bugs and feature ideas:** open an [issue](https://github.com/eurocontrol-asu/deepcontrail/issues)
  first, so we can agree on the change before you spend time on it.
- **Security problems:** do not open a public issue; see [SECURITY.md](SECURITY.md).
- **Conduct:** everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Set up

The repository is a [uv](https://docs.astral.sh/uv/) workspace. Python 3.11 or newer.

```bash
git clone https://github.com/eurocontrol-asu/deepcontrail.git
cd deepcontrail
uv sync                    # all packages, the dev tools and the training scripts' dependencies
uv sync --group convert    # only if you work on the OpenContrails converter (TensorFlow)
```

The packages live in `packages/<name>/src/deepcontrail/<name>/` and share the `deepcontrail`
namespace; the converter is in `tools/opencontrails/`. Each package has its own `tests/`.

## Make a change

1. Create a branch from `main` with a prefix that says what it is: `feature/…`, `fix/…`,
   `docs/…`, `refactor/…`, `test/…`, `chore/…` or `ci/…`.
2. **Write the test first.** Add a test that fails without your change, then make it pass. Tests
   use tiny synthetic arrays (`np.zeros`, `np.arange`), never real dataset files or the network,
   and seed every random generator they use.
3. Keep the change focused: one concern per pull request.
4. Run the four checks; all must be green:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest
   ```

5. Open a pull request against `main` and fill in the template. CI runs the same checks, and
   also tests each package in an environment with only that package installed.

## Rules the code follows

- **`deepcontrail-core` never imports torch.** `import deepcontrail.core` must work with only
  numpy, Pillow, pydantic and PyYAML; CI checks it. `deepcontrail-instance` is torch-free too.
- **The dataset format is the boundary.** The library reads the format described in
  [`packages/core`](packages/core) and never hardcodes a sensor's band names, sizes or paths.
- **Public names are explicit:** each package's `__init__.py` lists them in `__all__`. Public
  functions have type annotations and NumPy-style docstrings.
- **Masks stay binary.** A transform that changes a mask must keep it in {0, 1} and at the same
  size, and a test must show it.
- **Model changes come with a number.** A change that affects training or prediction reports
  the Dice on a fixed validation set, with the seed used.
- **Dependencies:** do not add one without discussing it in the issue first.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/): a type, then a short summary
in the imperative mood.

```text
feat: add a sample-scene download helper
fix: keep masks binary after rotation
docs: explain the pixel shift in the core README
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`.

## Licence of contributions

DeepContrail is licensed under the [EUPL-1.2](LICENSE). By submitting a contribution, you agree
that it is licensed under the EUPL-1.2 as well, and that you have the right to submit it.
