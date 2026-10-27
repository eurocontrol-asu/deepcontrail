# deepcontrail-core

Shared, torch-free foundations of [DeepContrail](https://github.com/eurocontrol-asu/deepcontrail):
the dataset configuration, the spectral band keys, false-colour composites built from named
bands, the train/val/fold partition, and Dice/IoU metrics.

Depends only on numpy, Pillow, pydantic and PyYAML. `import deepcontrail.core` never imports torch.
