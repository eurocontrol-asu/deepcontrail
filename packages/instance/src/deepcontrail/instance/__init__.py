"""Instance segmentation of contrails: separate the contrails of a segmentation map.

Mannstein-style line detection and region growing turn a binary or probability mask into
one mask and one score per contrail. Torch-free; scikit-learn is needed only to train a
``ScoreModel`` (the ``[scores]`` extra).
"""

from importlib.metadata import version

from deepcontrail.instance.config import InstancePipelineConfig, MannsteinConfig
from deepcontrail.instance.io import (
    instance_masks_from_annotators,
    instance_masks_from_label_image,
    to_coco_results,
)
from deepcontrail.instance.mannstein import detect_instances
from deepcontrail.instance.pipeline import detect_instances_from_probabilities
from deepcontrail.instance.refine import grow_instances, probability_weighted_scores
from deepcontrail.instance.scoring import (
    FEATURES,
    ScoreModel,
    instance_features,
    match_quality,
    training_data,
)

__version__ = version("deepcontrail-instance")

__all__ = [
    "FEATURES",
    "InstancePipelineConfig",
    "MannsteinConfig",
    "ScoreModel",
    "__version__",
    "detect_instances",
    "detect_instances_from_probabilities",
    "grow_instances",
    "instance_features",
    "instance_masks_from_annotators",
    "instance_masks_from_label_image",
    "match_quality",
    "probability_weighted_scores",
    "to_coco_results",
    "training_data",
]
