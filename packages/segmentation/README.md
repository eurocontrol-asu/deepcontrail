# deepcontrail-segmentation

Semantic segmentation of contrails in GOES-16 imagery, part of
[DeepContrail](https://github.com/eurocontrol-asu/deepcontrail): the segmentation data config and
dataset, transforms, model factory (segmentation-models-pytorch + timm encoders), losses, the
Lightning module, D4 test-time augmentation and evaluation.

`scripts/` (training and evaluation) and `configs/experiment.yaml` (the reference configuration)
live next to the package in the repository and are not part of the installed package.
