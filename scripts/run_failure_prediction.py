#!/usr/bin/env python3
"""Predict MedSAM failures from output-shape and activation-difference features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from counterfactual_medsam.manifest import parse_box_xyxy


DEFAULT_REMOTE_PROJECT_PREFIX = "/path/to/project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--original-metrics", required=True, type=Path)
    parser.add_argument("--activation-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--failure-metric", default="dice", choices=["dice", "boundary_f1", "hd95"])
    parser.add_argument("--failure-quantile", type=float, default=0.5)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument(
        "--remote-project-prefix",
        default=DEFAULT_REMOTE_PROJECT_PREFIX,
        help="Optional prefix to strip from absolute paths when reading cluster-produced CSVs.",
    )
    return parser.parse_args()


def resolve_path(value: str, project_root: Path, remote_project_prefix: str) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    text = str(value)
    if remote_project_prefix and text.startswith(remote_project_prefix):
        candidate = project_root / text.removeprefix(remote_project_prefix).lstrip("/")
        if candidate.exists():
            return candidate
    candidate = project_root / text
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def mask_area_fraction(path: Path) -> float:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    return float(mask.mean())


def box_features(box_value: str, shape_source_path: Path) -> dict[str, float]:
    x0, y0, x1, y1 = parse_box_xyxy(box_value)
    width, height = Image.open(shape_source_path).size
    box_width = max(0, x1 - x0)
    box_height = max(0, y1 - y0)
    image_area = max(1, width * height)
    return {
        "box_area_fraction": float((box_width * box_height) / image_area),
        "box_width_fraction": float(box_width / max(1, width)),
        "box_height_fraction": float(box_height / max(1, height)),
    }


def build_output_features(manifest: pd.DataFrame, metrics: pd.DataFrame, project_root: Path, args) -> pd.DataFrame:
    merged = manifest.merge(
        metrics[[args.case_column, "prediction_path"]],
        on=args.case_column,
        how="inner",
    )
    rows = []
    for _, row in merged.iterrows():
        prediction_path = resolve_path(str(row["prediction_path"]), project_root, args.remote_project_prefix)
        features = box_features(str(row[args.box_column]), prediction_path)
        pred_area = mask_area_fraction(prediction_path)
        features.update(
            {
                args.case_column: row[args.case_column],
                "pred_area_fraction": pred_area,
                "pred_to_box_area": pred_area / max(features["box_area_fraction"], 1e-8),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def build_activation_features(summary: pd.DataFrame, case_column: str) -> pd.DataFrame:
    value_columns = [
        "mean_abs_diff",
        "max_abs_diff",
        "boundary_mean_abs_diff",
        "interior_mean_abs_diff",
        "background_mean_abs_diff",
    ]
    pivot = summary.pivot(index=case_column, columns="layer", values=value_columns)
    pivot.columns = [f"{value}__{layer}" for value, layer in pivot.columns]
    return pivot.reset_index().fillna(0.0)


def failure_labels(metrics: pd.DataFrame, metric: str, quantile: float, case_column: str) -> tuple[pd.Series, float]:
    threshold = float(metrics[metric].quantile(quantile))
    if metric == "hd95":
        labels = metrics[metric] >= threshold
    else:
        labels = metrics[metric] <= threshold
    labels.index = metrics[case_column]
    return labels.astype(int), threshold


def evaluate_feature_set(
    name: str,
    frame: pd.DataFrame,
    labels: pd.Series,
    case_column: str,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
) -> dict[str, float | str | int]:
    frame = frame.set_index(case_column).loc[labels.index]
    x = frame.to_numpy(dtype=float)
    y = labels.to_numpy(dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    probabilities = np.zeros_like(y, dtype=float)
    counts = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in splitter.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(x[train_idx], y[train_idx])
        probabilities[test_idx] += model.predict_proba(x[test_idx])[:, 1]
        counts[test_idx] += 1

    probabilities = probabilities / np.maximum(counts, 1.0)
    predictions = probabilities >= 0.5
    return {
        "feature_set": name,
        "n_cases": int(len(y)),
        "n_features": int(x.shape[1]),
        "positive_cases": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
    }


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    metrics = pd.read_csv(args.original_metrics)
    activation_summary = pd.read_csv(args.activation_summary)

    labels, threshold = failure_labels(
        metrics,
        metric=args.failure_metric,
        quantile=args.failure_quantile,
        case_column=args.case_column,
    )
    output_features = build_output_features(manifest, metrics, project_root, args)
    activation_features = build_activation_features(activation_summary, args.case_column)
    combined_features = output_features.merge(activation_features, on=args.case_column, how="inner")

    output_feature_columns = [args.case_column, "box_area_fraction", "box_width_fraction", "box_height_fraction", "pred_area_fraction", "pred_to_box_area"]
    results = [
        evaluate_feature_set(
            "output_shape",
            output_features[output_feature_columns],
            labels,
            args.case_column,
            args.cv_splits,
            args.cv_repeats,
            args.random_state,
        ),
        evaluate_feature_set(
            "activation_diff",
            activation_features,
            labels,
            args.case_column,
            args.cv_splits,
            args.cv_repeats,
            args.random_state,
        ),
        evaluate_feature_set(
            "output_plus_activation",
            combined_features,
            labels,
            args.case_column,
            args.cv_splits,
            args.cv_repeats,
            args.random_state,
        ),
    ]

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(args.output_dir / "failure_prediction_summary.csv", index=False)

    labels_df = metrics[[args.case_column, args.failure_metric]].copy()
    labels_df["failure_label"] = labels_df[args.case_column].map(labels)
    labels_df["failure_metric"] = args.failure_metric
    labels_df["failure_threshold"] = threshold
    labels_df.to_csv(args.output_dir / "failure_prediction_labels.csv", index=False)

    print(results_df.to_string(index=False))
    print(f"Wrote failure prediction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
