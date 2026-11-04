# deepcontrail-instance

Instance segmentation of contrails, part of
[DeepContrail](https://github.com/eurocontrol-asu/deepcontrail): it splits a contrail
segmentation map into individual contrails, one mask and one score each.

It needs only numpy, scipy, pydantic and PyYAML (no torch) and never imports
`deepcontrail.segmentation`: any model's probability map works as input. Two optional extras:
`[map]` (pycocotools) to save detections as COCO results, and `[scores]` (scikit-learn) to train a
score model.

```bash
pip install "deepcontrail-instance[map,scores] @ git+https://github.com/eurocontrol-asu/deepcontrail#subdirectory=packages/instance"
```

## Usage

```python
import numpy as np
from deepcontrail.instance import detect_instances_from_probabilities

probs = np.load("scene_probs.npy")  # (H, W) probabilities in [0, 1]
masks, scores = detect_instances_from_probabilities(probs)
# masks: one boolean (H, W) mask per contrail; scores: confidence in [0, 1]
```

The pipeline thresholds the map (at 0.465), splits the mask into straight contrails with the
Mannstein algorithm, grows each contrail by one pixel into neighbours with probability above
0.371 (annotators draw contrails wider than the thresholded mask), and weights each score by the
contrail's mean probability. Its settings live in `InstancePipelineConfig`; the defaults equal
[`configs/instance_pipeline.yaml`](configs/instance_pipeline.yaml), and
`InstancePipelineConfig.consensus()` equals
[`configs/instance_pipeline_consensus.yaml`](configs/instance_pipeline_consensus.yaml), for when
only contrails agreed by most annotators matter. `detect_instances(binary_mask)` runs plain
Mannstein on any binary mask (settings: `MannsteinConfig`, whose defaults are the pipeline's;
[`configs/mannstein_annotator_masks.yaml`](configs/mannstein_annotator_masks.yaml) holds the
settings for annotators' own masks).

Mannstein's oriented line filter (`MannsteinConfig(line_filter=True)`) is off in the shipped
settings: on these masks it kept every pixel and took most of the run time.

**Input.** A `(H, W)` map of probabilities in [0, 1] (after the sigmoid, not logits), on the same
pixel grid as the masks it is compared with. NaN or infinite values are rejected. A binary mask
also works as input; growth and probability weighting then change nothing.

**Known limits.**
- All settings are in pixels, tuned on GOES-16 / OpenContrails scenes (about 2 km per pixel,
  contrails 1–3 px wide). Other resolutions need re-tuning.
- The cut (0.465) and the growth (into p > 0.371) suit DeepContrail's reference semantic model,
  trained on soft labels. For another model, re-tune the pipeline on that model's predictions for
  scenes it never trained on; tuning needs instance annotations.
- Two contrails crossing at a small angle come out as one: at 10° always, at 20° with the
  consensus settings. Touching, nearly parallel contrails also merge.
- The COCO mAP keeps at most 100 detections per sample (a warning says when a sample has more).

## Annotations and results

The annotation tools need only `instance_masks`: each annotator's mask, the agreement map, the
consensus mask and the consensus contrails (drawn by more than half of the annotators) are derived
from it by `deepcontrail.instance.labels`, for any number of annotators, one included.

Instance annotations use the `instance_masks/<split>/<id>.npy` format of the
[DeepContrail dataset format](../core): an `(N, H, W, A)` int8 array (1 = pixel of contrail n
drawn by annotator a, 0 = not, -1 = annotator a drew no contrail n; `A = 1` for one annotator),
read with `deepcontrail.core.readers.load_instance_masks`. `instance_masks_from_label_image` and
`instance_masks_from_annotators` build it from a label image or from per-annotator masks.

`to_coco_results(masks, scores, image_id)` (needs `[map]`) saves detections as COCO results with
RLE masks, so overlapping contrails stay separate:

```python
import json
from deepcontrail.instance import to_coco_results

json.dump(to_coco_results(masks, scores, image_id="scene_1"), open("instances.json", "w"))
```

## Learned scores

A `ScoreModel` predicts, from 20 features of each detected contrail (size, shape, probabilities,
surroundings), how well it matches the annotated contrails, and that prediction replaces its
score; the masks do not change. The features depend on the semantic model, so a score model is
valid only for the semantic model and pipeline settings it was trained with, and refuses other
pipeline settings. Training (`ScoreModel.fit`, with `training_data`) needs `[scores]`; a saved
model is plain JSON and predicts with numpy only:

```python
from deepcontrail.instance import (
    InstancePipelineConfig,
    ScoreModel,
    detect_instances_from_probabilities,
)

config = InstancePipelineConfig.from_yaml("configs/instance_pipeline.yaml")
model = ScoreModel.from_json("score_model.json")
masks, scores = detect_instances_from_probabilities(probs, config, score_model=model)
```

## Evaluation

`deepcontrail.instance.metrics.evaluate_annotator_samples` (needs `[map]`) scores detections
against each annotator's contrails as a separate ground truth: one sample per (scene, annotator)
pair. The headline is the COCO segmentation mAP averaged over IoU 0.25–0.75, pooled over all
samples; precision, recall and F1 are at IoU 0.5. Samples where the annotator drew no contrail are
skipped by default; `include_empty=True` keeps them, so every detection on them counts as false,
which is closer to real use.

Results on the OpenContrails 2025 test split (1,879 scenes, 2,700 annotator samples with at least
one contrail), from the probabilities of DeepContrail's reference semantic model (D4 TTA). Every
setting was chosen on validation scenes; the test split was only scored. The consensus columns
score each scene once against its consensus contrails.

| Settings | mAP | mAP, with empty | Precision (with empty) | Recall | F1 (with empty) | Consensus mAP | Consensus mAP, all scenes |
|---|---|---|---|---|---|---|---|
| `configs/instance_pipeline.yaml` (defaults) | 0.272 | 0.267 | 0.516 (0.489) | 0.382 | 0.439 (0.429) | 0.418 | 0.409 |
| `configs/instance_pipeline_consensus.yaml` (no growth) | 0.250 | 0.246 | 0.499 (0.474) | 0.346 | 0.408 (0.400) | 0.434 | 0.426 |
| defaults + learned scores | 0.277 | 0.270 | 0.516 (0.489) | 0.382 | 0.439 (0.429) | 0.428 | 0.415 |
| consensus settings + learned scores | 0.256 | 0.250 | 0.499 (0.474) | 0.346 | 0.408 (0.400) | 0.445 | 0.433 |

On each annotator's own mask instead of a prediction, the instance step alone reaches mAP 0.921
(`configs/mannstein_annotator_masks.yaml`). Most errors therefore come from the semantic step and
from contrail shape: the AP is 0.51 at IoU 0.25 but 0.02 at IoU 0.75. Annotators also disagree
strongly on individual contrails: one annotator scored against the others reaches F1 ≈ 0.35 on
validation scenes.

## Scripts (repository only)

Command-line tools in [`scripts/`](scripts), run from the repository root. They read only
`instance_masks/<split>/` from a dataset, plus one `<id>.npy` probability map per scene for a
semantic model's predictions.

- **`evaluate_instances.py`**: the evaluation above, on predictions (`--pred-dir`), on the
  consensus mask (`--use-gt-semantic`) or on each annotator's mask (`--use-annotator-semantic`);
  `--score-model`, `--include-empty`, `--consensus-gt`.
- **`train_score_model.py`**: trains a `ScoreModel` (needs `[scores]`).
- **`optimize_pipeline.py`** and **`optimize_mannstein.py`**: Optuna searches of the pipeline and
  Mannstein settings (optuna is in the repository's `experiments` group); they never read the
  test split.

Tuning for your own semantic model: predict the scenes it never trained on (for example its
validation split), search the pipeline settings on them (`optimize_pipeline.py`), train a score
model on the same predictions (`train_score_model.py`), then score the test split once
(`evaluate_instances.py`).

```bash
uv run python packages/instance/scripts/train_score_model.py --root <dataset_root> \
    --pred-dir <validation probability maps> \
    --pipeline-config packages/instance/configs/instance_pipeline.yaml --output-dir <dir>
```

## References and acknowledgements

The line filter and the size, length and straightness tests follow Mannstein, Meyer & Wendling
(1999), "Operational detection of contrails from NOAA-AVHRR data", *International Journal of
Remote Sensing* 20(8), 1641–1660, [doi:10.1080/014311699212650](https://doi.org/10.1080/014311699212650).

The step that separates crossing contrails (region growing that follows each contrail's direction)
is adapted from the PIE project at ISAE-SUPAERO, carried out with SII Research and EUROCONTROL by
J. Jault, A. Berrada-Gomez, A. Tonon, A. Maurette, T. Muffang and N. Grimmer.
