# deepcontrail

[DeepContrail](https://github.com/eurocontrol-asu/deepcontrail) is an open-source library for
contrail detection in satellite imagery. This package has no code of its own: it installs the
whole library, released together with it.

```bash
pip install deepcontrail
```

installs:

| Package | Import | Contents |
|---|---|---|
| [`deepcontrail-core`](https://pypi.org/project/deepcontrail-core/) | `deepcontrail.core` | The dataset format and its validation, false-colour composites, splits, Dice/IoU (torch-free) |
| [`deepcontrail-segmentation`](https://pypi.org/project/deepcontrail-segmentation/) | `deepcontrail.segmentation` | Semantic segmentation: training, pretrained models, prediction |
| [`deepcontrail-instance`](https://pypi.org/project/deepcontrail-instance/) (with `[map,scores]`) | `deepcontrail.instance` | Instance segmentation: individual contrails with learned scores, COCO results |

Each package can also be installed alone. The OpenContrails converter,
[`deepcontrail-opencontrails`](https://pypi.org/project/deepcontrail-opencontrails/), is a separate
tool (it needs TensorFlow) and is not included.

The quickstart, results and documentation are in the
[repository README](https://github.com/eurocontrol-asu/deepcontrail#readme).
