"""Part C: fit the baseline on development and generate the required test predictions.

Validation is held out from fitting. Validation metrics belong to Part D.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from part_a import load_training_data
from part_b import FEATURE_COLUMNS, FEATURE_SETTINGS, build_feature_table, load_image_rgb
from run_options import parse_run_options


# C1 - one fixed baseline; settings are not selected using test data.
MODEL_SETTINGS = {
    "features": FEATURE_COLUMNS,
    "iterations": 2000,
    "l2_penalty": 0.01,
    "threshold": 0.5,
    "solver": "lbfgs",
}
SPLIT_SETTINGS = {"validation_fraction": 0.25, "random_seed": 42}
EXPECTED_TEST_RECORDS = 72
PREDICTION_COLUMNS = ["image_id", "predicted_defect", "score"]


# C1 - logistic regression with development-only standardization and L2 regularization.
class LogisticBaseline:
    """StandardScaler and scikit-learn logistic regression for two image features.

    The score is a logistic model output, not a verified or calibrated defect
    probability. Metadata and annotation fields are never model inputs.
    """

    def __init__(self) -> None:
        self.pipeline_: Pipeline | None = None

    @staticmethod
    def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
        missing = set(FEATURE_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Model features must be non-empty and finite")
        return values

    def fit(self, development: pd.DataFrame, labels: np.ndarray) -> "LogisticBaseline":
        """Learn scaling and coefficients using development records only."""
        values = self._feature_matrix(development)
        target = np.asarray(labels, dtype=np.float64)
        if target.shape != (len(values),) or set(np.unique(target)) != {0.0, 1.0}:
            raise ValueError("Development labels must align with rows and include both 0 and 1")

        # Objective: mean log loss + (l2_penalty / 2) * ||w||^2.
        # scikit-learn scales the L2 term by 1/(n*C), so C = 1/(n*l2_penalty).
        inverse_regularization = 1.0 / (len(target) * MODEL_SETTINGS["l2_penalty"])
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=inverse_regularization,
                solver=MODEL_SETTINGS["solver"],
                max_iter=MODEL_SETTINGS["iterations"],
                tol=1e-8,
            )),
        ])
        self.pipeline_.fit(values, target)
        return self

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the stored development scaling without refitting it."""
        if self.pipeline_ is None:
            raise RuntimeError("Fit the model on development data before prediction")
        # Both labels were checked in fit(); column 1 is the score for label 1.
        return self.pipeline_.predict_proba(self._feature_matrix(frame))[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return 1 for scores at or above the prespecified threshold."""
        return (self.predict_score(frame) >= MODEL_SETTINGS["threshold"]).astype(int)


# C2 - exact duplicates must be reviewed before a random image-level split.
def check_exact_duplicates(frame: pd.DataFrame, data_dir: Path) -> None:
    """Detect identical decoded RGB pixels; this does not detect near-duplicates."""
    seen = {}
    for row in frame.itertuples(index=False):
        image = load_image_rgb(data_dir / row.file)
        signature = hashlib.sha256(str(image.shape).encode("ascii") + image.tobytes()).hexdigest()
        if signature in seen:
            raise ValueError(
                f"Exact image duplicate: {row.image_id} and {seen[signature]}. "
                "Review duplicate grouping before splitting."
            )
        seen[signature] = row.image_id


# C2 - one fixed, stratified split; validation labels are used for stratification only.
def split_development_validation(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain class proportions without sharing image IDs between the subsets."""
    if features["image_id"].isna().any() or features["image_id"].duplicated().any():
        raise ValueError("Splitting requires non-missing, unique image IDs")
    development, validation = train_test_split(
        features,
        test_size=SPLIT_SETTINGS["validation_fraction"],
        random_state=SPLIT_SETTINGS["random_seed"],
        stratify=features["inspector_label"],
    )
    if set(development["image_id"]) & set(validation["image_id"]):
        raise ValueError("Development and validation image IDs overlap")
    if len(development) + len(validation) != len(features):
        raise ValueError("The split does not cover all labeled records")
    return development.reset_index(drop=True), validation.reset_index(drop=True)


# C3 - validate unlabeled test identities and paths without using annotation fields.
def load_test_data(path: Path) -> pd.DataFrame:
    """Read the supplied test metadata for inference only, preserving input order."""
    frame = pd.read_csv(path, dtype={"image_id": "string", "file": "string"})
    if not {"image_id", "file"}.issubset(frame.columns):
        raise ValueError("Test metadata must contain image_id and file")
    if len(frame) != EXPECTED_TEST_RECORDS:
        raise ValueError(f"Expected {EXPECTED_TEST_RECORDS} test records; found {len(frame)}")
    for column in ["image_id", "file"]:
        if frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"Missing or blank test values in {column}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Test image IDs must be unique")
    return frame


# C3 - check the exact submission schema, row identity, score range and decision rule.
def validate_predictions(predictions: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Validate generated or reloaded predictions against the actual test ID list."""
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {PREDICTION_COLUMNS}")
    if len(predictions) != len(expected_ids) or predictions["image_id"].tolist() != expected_ids.tolist():
        raise ValueError("Prediction rows must match test IDs exactly, in the same order")
    if predictions["image_id"].isna().any() or predictions["image_id"].duplicated().any():
        raise ValueError("Prediction IDs must be non-missing and unique")
    scores = predictions["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be finite and between 0 and 1")
    if not predictions["predicted_defect"].isin([0, 1]).all():
        raise ValueError("Predicted defects must be 0 or 1")
    if not np.array_equal(predictions["predicted_defect"].to_numpy(),
                          (scores >= MODEL_SETTINGS["threshold"]).astype(int)):
        raise ValueError("Predictions do not match the fixed score threshold")


def main() -> None:
    options = parse_run_options("part_c", "Part C: split, fit on development, and predict test images.")
    options.output_dir.mkdir(parents=True, exist_ok=True)

    # C2 - rebuild pixel features from the source images, avoiding stale cached tables.
    train = load_training_data(options.data_dir / "train.csv")
    check_exact_duplicates(train, options.data_dir)
    features = build_feature_table(train, options.data_dir)
    development, validation = split_development_validation(features)
    development.to_csv(options.output_dir / "development_features.csv", index=False)
    validation.to_csv(options.output_dir / "validation_features.csv", index=False)
    assignments = pd.concat([
        development[["image_id", "inspector_label"]].assign(subset="development"),
        validation[["image_id", "inspector_label"]].assign(subset="validation"),
    ], ignore_index=True)
    assignments.sort_values("image_id").to_csv(options.output_dir / "split_assignments.csv", index=False)
    split_summary = pd.crosstab(assignments["subset"], assignments["inspector_label"]).reindex(
        index=["development", "validation"], columns=[0, 1], fill_value=0,
    )
    split_summary.columns = ["labeled_normal", "labeled_defective"]
    split_summary.insert(0, "image_count", split_summary.sum(axis=1))
    split_summary["labeled_defective_percentage"] = (
        split_summary["labeled_defective"] / split_summary["image_count"] * 100
    )
    split_summary.to_csv(options.output_dir / "split_summary.csv")

    # C1 after C2 - fit both preprocessing and the classifier on development only.
    model = LogisticBaseline().fit(development, development["inspector_label"].to_numpy())
    model_path = options.output_dir / "development_pipeline.joblib"
    joblib.dump(model.pipeline_, model_path)
    specification = {"model": "L2-regularized logistic regression", **MODEL_SETTINGS,
                     "implementation": "scikit-learn StandardScaler + LogisticRegression Pipeline",
                     "regularization_mapping": "C = 1 / (development_record_count * l2_penalty)",
                     "status": "Fitted on development only; validation is evaluated separately in Part D"}
    (options.output_dir / "baseline_specification.json").write_text(
        json.dumps(specification, indent=2) + "\n", encoding="utf-8",
    )
    run_config = {
        **SPLIT_SETTINGS,
        "development_records": len(development),
        "validation_records": len(validation),
        "feature_columns": FEATURE_COLUMNS,
        "feature_definition": "Mean absolute differences from left/right and top/bottom reflections of the aligned, normalized panel",
        "feature_settings": FEATURE_SETTINGS,
        "scikit_learn_version": sklearn.__version__,
        "classifier_C": model.pipeline_.named_steps["classifier"].C,
        "exact_pixel_duplicates": 0,
        "split_scope": "Random image-level split; batches and cameras may appear in both subsets",
        "limitation": "Exploratory Parts A-B used all labeled records; validation is held out from fitting, not prior exploration",
    }
    (options.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8",
    )
    print("\nPart C2 - A fixed development and validation split")
    print("We split labeled images 75/25 with seed=42, keeping the class proportions in both subsets.")
    print("No exact decoded-image duplicates were found. Each image ID belongs to one subset only.")
    print(split_summary.to_string(float_format=lambda value: f"{value:.2f}"))
    print("\nThe scaler and logistic model were fitted using development records only.")
    print("The threshold remains 0.5. Validation metrics will be computed in Part D; test inference follows in C3.")
    print("This image-level split does not measure performance on an entirely new batch or camera.")
    print("Earlier exploration covered all labeled records; the validation subset was not used for model fitting.")
    print(f"Saved split records and settings in: {options.output_dir}")
    print(f"Saved fitted development pipeline: {model_path}")

    # C3 - read test inputs only after the development model and settings are fixed.
    test = load_test_data(options.data_dir / "test.csv")
    if set(test["image_id"]) & set(train["image_id"]):
        raise ValueError("Test image IDs overlap with labeled training image IDs")
    test_features = build_feature_table(test, options.data_dir)

    # Reload the saved development pipeline; do not fit anything on the test inputs.
    inference_model = LogisticBaseline()
    inference_model.pipeline_ = joblib.load(model_path)
    scores = inference_model.predict_score(test_features)
    predictions = pd.DataFrame({
        "image_id": test["image_id"].to_numpy(),
        "predicted_defect": (scores >= MODEL_SETTINGS["threshold"]).astype(int),
        "score": scores,
    }, columns=PREDICTION_COLUMNS)
    validate_predictions(predictions, test["image_id"])
    predictions_path = options.output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    # Read the delivered CSV back to check its actual contents, not only the in-memory table.
    delivered = pd.read_csv(predictions_path, dtype={"image_id": "string"})
    validate_predictions(delivered, test["image_id"])
    test_features.to_csv(options.output_dir / "test_image_features.csv", index=False)

    print("\nPart C3 - Predictions for the unlabeled test images")
    print(f"Saved one prediction for each of the {len(predictions)} test images, in the original order.")
    print(f"Predicted suspicious: {int(predictions['predicted_defect'].sum())}; "
          f"predicted normal: {int((predictions['predicted_defect'] == 0).sum())}.")
    print(f"Scores range from {scores.min():.3f} to {scores.max():.3f}; the decision threshold is 0.5.")
    print("Test images were used for feature extraction and inference only, not fitting or tuning.")
    print("These are model predictions, not verified test outcomes. Run Part D for validation evaluation.")
    print(f"Submission file: {predictions_path}")


if __name__ == "__main__":
    main()
