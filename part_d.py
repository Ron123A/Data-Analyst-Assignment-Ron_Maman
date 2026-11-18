"""Part D: validation metrics and one reproducibly selected case for review.

Run Part C first, using the same output root. This script never fits a model,
changes the decision threshold, or reads the unlabeled test data.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

from part_a import load_training_data
from part_b import FEATURE_COLUMNS, FEATURE_SETTINGS, load_image_rgb, prepare_panel
from run_options import parse_run_options


# D1 - confirm that the saved evaluation records really belong to validation.
def load_validation_artifacts(artifact_dir: Path, data_dir: Path) -> tuple:
    """Read the saved split and pipeline, checking identity and label alignment."""
    filenames = ["development_pipeline.joblib", "validation_features.csv",
                 "split_assignments.csv", "baseline_specification.json", "run_config.json"]
    missing = [name for name in filenames if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run Part C first with the same output root. Missing: {missing}")
    validation = pd.read_csv(artifact_dir / "validation_features.csv")
    assignments = pd.read_csv(artifact_dir / "split_assignments.csv")
    specification = json.loads((artifact_dir / "baseline_specification.json").read_text(encoding="utf-8"))
    configuration = json.loads((artifact_dir / "run_config.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(artifact_dir / "development_pipeline.joblib")
    source = load_training_data(data_dir / "train.csv").set_index("image_id")

    expected_ids = set(assignments.loc[assignments["subset"] == "validation", "image_id"])
    development_ids = set(assignments.loc[assignments["subset"] == "development", "image_id"])
    if validation.empty or validation["image_id"].isna().any() or validation["image_id"].duplicated().any():
        raise ValueError("Validation records must have non-missing, unique image IDs")
    if set(validation["image_id"]) != expected_ids or expected_ids & development_ids:
        raise ValueError("Validation records do not match the saved, disjoint split")
    if len(validation) != configuration["validation_records"]:
        raise ValueError("Validation count does not match the saved run configuration")
    if not expected_ids.issubset(source.index):
        raise ValueError("Saved validation IDs are not present in the supplied train.csv")
    if not np.array_equal(source.loc[validation["image_id"], "inspector_label"].to_numpy(),
                          validation["inspector_label"].to_numpy()):
        raise ValueError("Saved validation labels differ from the supplied train.csv")
    if specification["features"] != FEATURE_COLUMNS:
        raise ValueError("Saved model feature definitions do not match the evaluation inputs")
    if configuration.get("feature_settings") != FEATURE_SETTINGS:
        raise ValueError("Saved preprocessing settings differ from the current feature function")
    if validation["file"].tolist() != source.loc[validation["image_id"], "file"].tolist():
        raise ValueError("Saved validation image paths differ from the supplied train.csv")
    x = validation[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Validation features must be finite")
    if pipeline.named_steps["scaler"].n_samples_seen_ != configuration["development_records"]:
        raise ValueError("The scaler fit count differs from the saved development size")
    threshold = float(specification["threshold"])
    if threshold != 0.5:
        raise ValueError("This evaluation expects the prespecified threshold of 0.5")
    return validation, pipeline, threshold


# D1 - evaluate the defective class (1), preserving undefined metrics as missing.
def evaluate_validation(validation: pd.DataFrame, pipeline, threshold: float) -> tuple:
    """Predict with the stored scaler and classifier, without any fit or tuning."""
    x = validation[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    classes = pipeline.named_steps["classifier"].classes_
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Saved model has no unique positive class labeled 1")
    scores = pipeline.predict_proba(x)[:, positive_columns[0]]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Validation scores must be finite and in [0, 1]")
    predicted = (scores >= threshold).astype(int)
    actual = validation["inspector_label"].to_numpy(dtype=int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = float(precision_score(actual, predicted, pos_label=1, zero_division=np.nan))
    recall = float(recall_score(actual, predicted, pos_label=1, zero_division=np.nan))
    auc_defined = len(np.unique(actual)) == 2
    auc = float(roc_auc_score(actual, scores)) if auc_defined else None
    metrics = {
        "validation_records": len(validation), "positive_label": 1, "threshold": threshold,
        "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
        "precision": None if np.isnan(precision) else precision,
        "recall": None if np.isnan(recall) else recall,
        "roc_auc": auc,
        "precision_defined": tp + fp > 0, "recall_defined": tp + fn > 0,
        "roc_auc_defined": auc_defined,
    }
    predictions = validation.copy()
    predictions["predicted_defect"] = predicted
    predictions["score"] = scores
    return matrix, metrics, predictions


# D1 - show inspector labels on rows and model predictions on columns.
def plot_confusion(matrix: np.ndarray, output_dir: Path, threshold: float, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    maximum = max(1, int(matrix.max()))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
    for row, column, name in [(0, 0, "TN"), (0, 1, "FP"), (1, 0, "FN"), (1, 1, "TP")]:
        count = int(matrix[row, column])
        ax.text(column, row, f"{name}\n{count}", ha="center", va="center", fontsize=18,
                color="white" if count > maximum / 2 else "#172B3A")
    labels = ["Normal (0)", "Defective (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Inspector label")
    ax.set_title("Part D1 - Validation confusion matrix", pad=15)
    fig.text(0.12, 0.025, f"Validation only: {matrix.sum()} images. Fixed score threshold: {threshold}.",
             fontsize=9, color="#52616B")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "validation_confusion_matrix.png", dpi=180)
    if show:
        plt.show()
    plt.close(fig)


# D2 - choose one error without cherry-picking its appearance or changing the model.
def select_review_case(predictions: pd.DataFrame, threshold: float) -> pd.Series:
    """Select the error nearest the boundary; if none, select the nearest case."""
    errors = predictions.loc[predictions["inspector_label"] != predictions["predicted_defect"]]
    candidates = errors if not errors.empty else predictions
    return candidates.assign(distance=(candidates["score"] - threshold).abs()).sort_values(
        ["distance", "image_id"], kind="stable",
    ).iloc[0]


def review_validation_case(predictions: pd.DataFrame, data_dir: Path, output_dir: Path,
                           threshold: float, show: bool) -> None:
    """Save one original image, the measured panel, and a cautious explanation."""
    case = select_review_case(predictions, threshold)
    actual, predicted = int(case["inspector_label"]), int(case["predicted_defect"])
    is_error = actual != predicted
    selection = ("The misclassified validation image closest to the fixed threshold was selected."
                 if is_error else "There were no validation errors; the image closest to the threshold was selected.")
    if actual == 1 and predicted == 0:
        explanation = (
            "A small or nearly symmetric defect can leave the two average reflection differences low. "
            "Averaging across the panel can dilute a local mark, and cropping can remove damage at its outer edge. "
            "These are possible explanations for the missed inspector label, not a confirmed defect diagnosis."
        )
    elif actual == 0 and predicted == 1:
        explanation = (
            "The panel's lower lettering has no matching lettering above the upper row. "
            "Such normal design details contribute to top/bottom asymmetry; uneven illumination or "
            "imperfect alignment can raise the reflection differences further. "
            "The model can therefore flag an image even when the inspector labeled it normal. "
            "These are possible explanations for the disagreement, not proof that either label is correct."
        )
    else:
        explanation = (
            "The score is close to the decision boundary. Small changes in alignment or illumination could "
            "change the prediction, and two average symmetry measurements cannot resolve every local defect."
        )
    text = (
        f"Part D2 - One validation case\n{selection}\n"
        f"Image: {case['image_id']}; inspector label: {actual}; predicted label: {predicted}; "
        f"score: {case['score']:.4f}; threshold: {threshold:.2f}.\n"
        f"Left/right asymmetry: {case['horizontal_asymmetry']:.4f}; "
        f"top/bottom asymmetry: {case['vertical_asymmetry']:.4f}.\n{explanation}\n"
        "The score is a model output, not a calibrated probability. This review does not change the model or threshold.\n"
    )
    (output_dir / "validation_case_review.txt").write_text(text, encoding="utf-8")
    case.drop(labels=["distance"]).to_frame().T.to_csv(output_dir / "validation_case.csv", index=False)
    path = data_dir / case["file"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5))
    axes[0].imshow(load_image_rgb(path), interpolation="nearest")
    axes[0].set_title("Original image")
    axes[1].imshow(prepare_panel(path), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Aligned and normalized panel")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Part D2 - {case['image_id']} | Inspector: {actual} | Predicted: {predicted}", fontsize=13)
    fig.text(0.08, 0.025, f"Validation only. Score: {case['score']:.4f}; fixed threshold: {threshold:.2f}.",
             fontsize=10, color="#52616B")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_dir / "validation_case_review.png", dpi=180)
    print(f"\n{text}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    options = parse_run_options("part_d", "Part D: validation metrics and review of one image.")
    if options.no_show:
        plt.switch_backend("Agg")
    artifact_dir = options.output_dir.parent / "part_c"
    validation, pipeline, threshold = load_validation_artifacts(artifact_dir, options.data_dir)
    matrix, metrics, predictions = evaluate_validation(validation, pipeline, threshold)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(options.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(options.output_dir / "validation_metrics.csv", index=False)
    (options.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    pd.DataFrame(matrix, index=["Inspector normal (0)", "Inspector defective (1)"],
                 columns=["Predicted normal (0)", "Predicted defective (1)"]).to_csv(
        options.output_dir / "validation_confusion_matrix.csv", index_label="inspector_label",
    )
    print("\nPart D1 - How the fixed model performed on validation")
    print(f"We evaluated {len(validation)} held-out records using the saved model and threshold {threshold}.")
    print(f"Normal-labeled images predicted normal (TN): {metrics['true_negative']}")
    print(f"Normal-labeled images flagged as defective (FP): {metrics['false_positive']}")
    print(f"Defective-labeled images missed by the model (FN): {metrics['false_negative']}")
    print(f"Defective-labeled images detected by the model (TP): {metrics['true_positive']}")
    if metrics["precision_defined"]:
        print(f"Precision for label 1: {metrics['precision']:.2%}")
    else:
        print("Precision for label 1 is undefined: the model did not predict any defective images (TP + FP = 0).")
    if metrics["recall_defined"]:
        print(f"Recall for label 1: {metrics['recall']:.2%}")
    else:
        print("Recall for label 1 is undefined: there are no defective-labeled validation images.")
    if metrics["roc_auc_defined"]:
        print(f"ROC-AUC for label 1: {metrics['roc_auc']:.3f} (computed from scores, independent of the threshold).")
    else:
        print("ROC-AUC is undefined: both inspector classes must be present in validation.")
    if predictions["predicted_defect"].eq(0).all():
        print("At this threshold, the model behaves like an always-normal rule on this validation subset.")
    print("No model parameters or thresholds were changed. These results measure agreement with inspector labels.")
    print(f"Saved validation results in: {options.output_dir}")
    plot_confusion(matrix, options.output_dir, threshold, show=not options.no_show)
    review_validation_case(predictions, options.data_dir, options.output_dir, threshold, show=not options.no_show)


if __name__ == "__main__":
    main()
