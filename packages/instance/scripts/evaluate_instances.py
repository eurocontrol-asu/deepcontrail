"""Instance segmentation evaluation per annotator sample (pooled mAP over IoU 0.25-0.75).

Every (scene, annotator) pair is one sample: the predictions are scored against that
annotator's own instances. Samples where the annotator drew no contrail are skipped. All
samples are pooled into one COCO segmentation mAP averaged over IoU 0.25, 0.35, ..., 0.75
(the headline), with the AP at each threshold. Precision/recall/F1 from one-to-one
matching at ``--match-iou`` (0.5) are printed as a reference: they show precision directly,
while mAP counts false detections only through the ranking by score.

Only ``instance_masks/<split>/`` is read from the dataset, for any number of annotators (one
included): each annotator's mask and the consensus mask are derived from it.

``--include-empty`` also reports the same metrics with the empty samples kept (every detection
on them is false), which is closer to real use, where most scenes have no contrail.

``--consensus-gt`` (ceiling and end-to-end modes) also scores each scene once against a
consensus instance set built from the annotators (contrails drawn by more than half of them,
clipped to the consensus mask): a diagnostic of instancing without annotator disagreement,
printed next to the per-annotator metrics. With a single annotator, the two coincide.

Three input modes:

- ``--use-annotator-semantic``: each annotator's own semantic mask is the input (the
  per-annotator benchmark of the instance step alone; detection runs once per sample).
- ``--use-gt-semantic``: the consensus mask, pixels drawn by more than half of the annotators
  (algorithm ceiling).
- ``--pred-dir``: per-scene predictions of any semantic model, ``<stem>.npy`` probabilities
  in ``[0, 1]`` or ``<stem>.png`` masks, on the pixel grid of the masks (end-to-end).

The pipeline is ``--pipeline-config`` (optimised pipeline), or plain Mannstein
(``--mannstein-config``) on the map thresholded at ``--threshold``. ``--score-model``
replaces the pipeline's scores by a learned :class:`~deepcontrail.instance.ScoreModel` (see
``scripts/train_score_model.py``). On the OpenContrails export, the ``test`` split is
OpenContrails' held-out validation split.

Usage
-----
::

    R=<dataset_root>
    P=<folder of per-scene probability maps>

    # Instance step alone, on each annotator's own mask
    uv run python packages/instance/scripts/evaluate_instances.py --root $R --use-annotator-semantic
        --mannstein-config packages/instance/configs/mannstein_annotator_masks.yaml

    # End-to-end with the default pipeline
    uv run python packages/instance/scripts/evaluate_instances.py --root $R --pred-dir $P
        --pipeline-config packages/instance/configs/instance_pipeline.yaml
        --consensus-gt --include-empty
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from deepcontrail.core.readers import load_instance_masks
from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig, ScoreModel
from deepcontrail.instance.evaluation import (
    Sample,
    build_samples,
    check_size,
    default_workers,
    detect_all,
    detect_all_masks,
    list_scenes,
    load_prediction,
    prediction_path,
)
from deepcontrail.instance.labels import consensus_instances, consensus_mask_from_instances
from deepcontrail.instance.metrics import evaluate_annotator_samples

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.5


def build_pipeline_config(args: argparse.Namespace) -> InstancePipelineConfig:
    """Return the pipeline config selected by the command-line options.

    ``--pipeline-config`` loads a full ``InstancePipelineConfig``. Otherwise the baseline
    is built: plain Mannstein (``--mannstein-config`` or defaults) on the map thresholded at
    ``--threshold``, without growth and with Mannstein's own scores.
    """
    if args.pipeline_config:
        return InstancePipelineConfig.from_yaml(args.pipeline_config)
    mannstein = (
        MannsteinConfig.from_yaml(args.mannstein_config)
        if args.mannstein_config
        else MannsteinConfig()
    )
    threshold = _DEFAULT_THRESHOLD if args.threshold is None else args.threshold
    return InstancePipelineConfig(
        semantic_threshold=threshold,
        grow_steps=0,
        probability_scores=False,
        mannstein=mannstein,
    )


def _run(args: argparse.Namespace, pipeline_cfg: InstancePipelineConfig) -> dict[str, Any]:
    """Build the samples, run the pipeline, score per annotator sample and report."""
    instance_dir = Path(args.root) / "instance_masks" / args.split
    stems = list_scenes(args.root, args.split)
    if args.limit is not None:
        stems = stems[: args.limit]

    samples: list[Sample] = []
    gt_inputs: list[np.ndarray] = []
    shapes: dict[str, tuple[int, ...]] = {}
    consensus: list[list[np.ndarray]] = []
    for stem in stems:
        raw = load_instance_masks(instance_dir / f"{stem}.npy")
        shapes[stem] = raw.shape[1:3]
        samples.extend(build_samples(raw, stem))
        if args.use_gt_semantic:
            gt_inputs.append(consensus_mask_from_instances(raw).astype(np.float32))
        if args.consensus_gt:
            consensus.append(consensus_instances(raw))

    if args.use_annotator_semantic:
        mode = "per-annotator benchmark (each annotator's own semantic mask)"
        detections = detect_all_masks(
            [s.annotator_mask for s in samples], pipeline_cfg.mannstein, args.workers
        )
        preds = [d[0] for d in detections]
        scores = [d[1] for d in detections]
    else:
        if args.pred_dir is not None:
            mode = "end-to-end (predicted masks)"
            missing = [s for s in stems if not prediction_path(args.pred_dir, s).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(stems)} stems have no prediction file in "
                    f"{args.pred_dir}: " + ", ".join(missing[:5])
                )
            inputs = []
            for stem in stems:
                prediction = load_prediction(prediction_path(args.pred_dir, stem))
                check_size(prediction, shapes[stem], stem)
                inputs.append(prediction)
        else:
            mode = "algorithm ceiling (GT consensus mask)"
            inputs = gt_inputs
        score_model = ScoreModel.from_json(args.score_model) if args.score_model else None
        per_scene = dict(
            zip(stems, detect_all(inputs, pipeline_cfg, args.workers, score_model), strict=True)
        )
        preds = [per_scene[s.stem][0] for s in samples]
        scores = [per_scene[s.stem][1] for s in samples]

    gts = [s.gt for s in samples]
    metrics = evaluate_annotator_samples(
        preds, gts, None if args.uniform_scores else scores, match_iou=args.match_iou
    )
    evaluated = [p for p, g in zip(preds, gts, strict=True) if g]
    result: dict[str, Any] = {
        "mode": mode,
        "split": args.split,
        "pipeline_config": args.pipeline_config,
        "mannstein_config": args.mannstein_config,
        "score_model": args.score_model,
        "semantic_threshold": pipeline_cfg.semantic_threshold,
        "grow_steps": pipeline_cfg.grow_steps,
        "uniform_scores": args.uniform_scores,
        "n_scenes": len(stems),
        "n_annotator_samples": len(samples),
        "n_pred_instances_per_evaluated_sample": (
            float(np.mean([len(p) for p in evaluated])) if evaluated else 0.0
        ),
        "n_gt_instances_per_evaluated_sample": (
            float(np.mean([len(g) for g in gts if g])) if evaluated else 0.0
        ),
        **metrics,
    }
    if args.include_empty:
        result["with_empty"] = evaluate_annotator_samples(
            preds,
            gts,
            None if args.uniform_scores else scores,
            match_iou=args.match_iou,
            include_empty=True,
        )
    if args.consensus_gt:
        result["consensus"] = {
            **evaluate_annotator_samples(
                [per_scene[stem][0] for stem in stems],
                consensus,
                None if args.uniform_scores else [per_scene[stem][1] for stem in stems],
                match_iou=args.match_iou,
            ),
            "n_instances": sum(len(c) for c in consensus),
        }
        if args.include_empty:
            result["consensus"]["with_empty"] = evaluate_annotator_samples(
                [per_scene[stem][0] for stem in stems],
                consensus,
                None if args.uniform_scores else [per_scene[stem][1] for stem in stems],
                match_iou=args.match_iou,
                include_empty=True,
            )
    _print_report(result, args)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return result


def _print_report(r: dict[str, Any], args: argparse.Namespace) -> None:
    mannstein_source = args.mannstein_config or ("-" if args.pipeline_config else "defaults")
    print("\n" + "=" * 64)
    print("INSTANCE SEGMENTATION EVALUATION (per annotator sample)")
    print("=" * 64)
    print(f"Mode:                  {r['mode']}")
    print(f"Split:                 {r['split']}")
    print(f"Pipeline config:       {args.pipeline_config or '-'}")
    print(f"Mannstein config:      {mannstein_source}")
    print(f"Semantic threshold:    {r['semantic_threshold']}   growth steps: {r['grow_steps']}")
    print(f"Scores:                {'uniform 1.0' if r['uniform_scores'] else 'pipeline scores'}")
    print(
        f"Scenes / samples:      {r['n_scenes']} / {r['n_annotator_samples']} "
        f"({r['n_samples']} evaluated, {r['n_skipped_empty']} empty skipped)"
    )
    print(f"mAP@[0.25:0.75]:       {r['map']:.4f}")
    ap = "  ".join(f"{t:.2f}: {v:.4f}" for t, v in r["ap_per_threshold"].items())
    print(f"AP per IoU threshold:  {ap}")
    print(
        f"Reference @IoU {args.match_iou}:   P {r['precision']:.4f} / R {r['recall']:.4f} / "
        f"F1 {r['f1']:.4f}  (TP {r['tp']}, FP {r['fp']}, FN {r['fn']})"
    )
    print(
        f"Instances per sample:  predicted {r['n_pred_instances_per_evaluated_sample']:.2f}, "
        f"GT {r['n_gt_instances_per_evaluated_sample']:.2f}"
    )
    if "consensus" in r:
        c = r["consensus"]
        print("-" * 64)
        print(
            f"Consensus GT:          mAP {c['map']:.4f}  (P {c['precision']:.4f} / "
            f"R {c['recall']:.4f} / F1 {c['f1']:.4f}; {c['n_instances']} majority contrails "
            f"in {c['n_samples']} scenes; diagnostic)"
        )
    if "with_empty" in r:
        e = r["with_empty"]
        print("-" * 64)
        print(
            f"With empty samples:    mAP {e['map']:.4f}  (P {e['precision']:.4f} / "
            f"R {e['recall']:.4f} / F1 {e['f1']:.4f}; {e['n_samples']} samples, FP {e['fp']})"
        )
    if "with_empty" in r.get("consensus", {}):
        e = r["consensus"]["with_empty"]
        print(
            f"Consensus, all scenes: mAP {e['map']:.4f}  (P {e['precision']:.4f} / "
            f"R {e['recall']:.4f} / F1 {e['f1']:.4f}; {e['n_samples']} scenes, FP {e['fp']})"
        )
    print("=" * 64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate contrail instances per annotator sample (pooled mAP)."
    )
    parser.add_argument("--root", required=True, help="Dataset root path.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Split folder (default: test = OpenContrails validation).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--use-annotator-semantic",
        action="store_true",
        help="Input = each annotator's own semantic mask (instance step alone).",
    )
    mode.add_argument(
        "--use-gt-semantic",
        action="store_true",
        help="Input = the consensus GT semantic mask (algorithm ceiling).",
    )
    mode.add_argument(
        "--pred-dir",
        default=None,
        help="Folder of per-scene predictions (<stem>.npy probabilities or <stem>.png masks).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Baseline mode: probability above which a pixel is contrail (default: 0.5).",
    )
    parser.add_argument(
        "--mannstein-config",
        default=None,
        help="Baseline mode: YAML with a mannstein: section (default: MannsteinConfig()).",
    )
    parser.add_argument(
        "--pipeline-config",
        default=None,
        help="InstancePipelineConfig YAML (pipeline: and mannstein: sections).",
    )
    parser.add_argument(
        "--score-model",
        default=None,
        help="ScoreModel JSON that re-scores the detections (one probability map per scene).",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU of the reference precision/recall/F1 (default: 0.5).",
    )
    parser.add_argument(
        "--uniform-scores",
        action="store_true",
        help="Rank all detections equally (score 1.0) instead of the pipeline scores.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also report the metrics with empty samples kept (their detections are false).",
    )
    parser.add_argument(
        "--consensus-gt",
        action="store_true",
        help=(
            "Also score against one consensus instance set per scene (contrails drawn by more "
            "than half of the annotators, clipped to the consensus mask). Diagnostic; not with "
            "--use-annotator-semantic."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Worker processes (default: up to 8, at most the CPU count).",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N scenes only.")
    parser.add_argument("--output", default=None, help="Also write the result as JSON here.")
    args = parser.parse_args(argv)
    if args.pipeline_config and (args.threshold is not None or args.mannstein_config):
        parser.error("--pipeline-config already sets the threshold and the Mannstein settings")
    if args.score_model and args.use_annotator_semantic:
        parser.error("--score-model needs one probability map per scene")
    if args.consensus_gt and args.use_annotator_semantic:
        parser.error("--consensus-gt needs one input per scene (--use-gt-semantic or --pred-dir)")
    return args


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Build the pipeline config and run the evaluation."""
    return _run(args, build_pipeline_config(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
