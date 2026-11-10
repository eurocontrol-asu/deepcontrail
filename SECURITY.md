# Security policy

## Supported versions

DeepContrail is in alpha. Only the latest release receives security fixes.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please do not report security problems in a public issue. Instead, either:

- use GitHub's [private vulnerability reporting](https://github.com/eurocontrol-asu/deepcontrail/security/advisories/new), or
- email [github-asu@eurocontrol.int](mailto:github-asu@eurocontrol.int).

Describe the problem, the affected version and, if you can, how to reproduce it. We will
acknowledge the report, keep you informed, and credit you in the fix unless you prefer not.

## What to keep in mind

- **Pretrained bundles are data the library parses.** `load_pretrained` reads `config.json`,
  `model.safetensors` and the instance-step files (`instance_pipeline.yaml`, `score_model.json`)
  and validates them; it does not run code from the bundle. Still, load bundles only from
  sources you trust, and pin a `revision`.
- **Training checkpoints** (`.ckpt`) are PyTorch pickle files. Load only checkpoints you created
  or trust; the released weights use safetensors, which hold no code.
