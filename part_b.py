"""Part B: inspect training images before choosing image features.

Implements B1 (visual examples), B2 (two image features), and B3 (training-set comparison).
Close the Matplotlib window to finish an interactive run.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part_a import load_training_data
from run_options import parse_run_options


# Setup - runtime paths are resolved in main(); sampling settings stay fixed.
RANDOM_SEED = 42
FEATURE_COLUMNS = ["horizontal_asymmetry", "vertical_asymmetry"]
FEATURE_SETTINGS = {
    "panel_size": 128,
    "border_crop_pixels": 6,
    "normalization_percentiles": [5, 95],
    "minimum_foreground_fraction": 0.1,
    "minimum_percentile_range": 1.0,
}
# B1 - observation written after visually reviewing the fixed selection below.
VISUAL_OBSERVATION = (
    "IMG_0013, which the inspector labeled defective, shows a dark triangular area "
    "interrupting the component's right edge. The three normal-labeled examples have "
    "continuous right edges. This difference is worth investigating, but six examples "
    "are not enough to establish a general rule or confirm the inspector's labels."
)


# Shared image reader - preserve color and fail visibly on unreadable inputs.
def load_image_rgb(path: Path) -> np.ndarray:
    """Read an image, supporting Windows paths, and return RGB uint8 pixels."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Empty image file: {path}")
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# B1 - select three examples per inspector class without visual cherry-picking.
def select_examples(train: pd.DataFrame) -> pd.DataFrame:
    required = {"file", "camera", "exposure_setting"}
    if not required.issubset(train.columns):
        raise ValueError(f"Missing image-display columns: {sorted(required.difference(train.columns))}")
    selected = []
    for label in [0, 1]:
        candidates = train.loc[train["inspector_label"] == label].sort_values("image_id")
        if len(candidates) < 3:
            raise ValueError(f"Need at least three training examples with label {label}")
        selected.append(candidates.sample(n=3, random_state=RANDOM_SEED))
    return pd.concat(selected, ignore_index=True)


# B1 - use original colors without per-image brightness or contrast adjustment.
def display_examples(examples: pd.DataFrame, data_dir: Path, output_dir: Path, show: bool = True) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (_, row) in zip(axes.flat, examples.iterrows()):
        image = load_image_rgb(data_dir / row["file"])
        description = "Labeled normal" if row["inspector_label"] == 0 else "Labeled defective"
        ax.imshow(image, interpolation="nearest")
        ax.set_title(
            f"{row['image_id']} - {description}\n"
            f"{row['camera']} | Exposure: {row['exposure_setting']}",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle("Part B1 - Six training examples by inspector label", fontsize=15)
    fig.text(0.04, 0.025,
             "Three examples per label, selected with seed=42 for comparison, not to represent class proportions.",
             fontsize=9, color="#52616B")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_dir / "six_training_examples.png", dpi=180)
    if show:
        plt.show()
    plt.close(fig)


# B2 - prepare each image independently; no labels or dataset statistics are used.
def prepare_panel(path: Path) -> np.ndarray:
    """Align the largest bright object, crop its border and normalize its pixels.

    This assumes a bright, approximately rectangular panel on a dark background.
    Otsu's threshold is computed separately for each image. The rectangle
    corrects in-plane rotation; it does not recover arbitrary perspective.
    All numerical settings are fixed choices for this dataset, not universal defaults.
    """
    rgb = load_image_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground component found in {path.name}")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < FEATURE_SETTINGS["minimum_foreground_fraction"] * gray.size:
        raise ValueError(f"Foreground component is too small in {path.name}")
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
    center = corners.mean(axis=0)
    # Order corners around the center, starting at the top-left corner.
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0).astype(np.float32)
    size = FEATURE_SETTINGS["panel_size"]
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    panel = cv2.warpPerspective(gray, transform, (size, size), flags=cv2.INTER_LINEAR).astype(np.float64)
    crop = FEATURE_SETTINGS["border_crop_pixels"]
    panel = panel[crop:-crop, crop:-crop]
    low, high = np.percentile(panel, FEATURE_SETTINGS["normalization_percentiles"])
    if high - low < FEATURE_SETTINGS["minimum_percentile_range"]:
        raise ValueError(f"Insufficient panel contrast in {path.name}")
    return np.clip((panel - low) / (high - low), 0, 1)


def measure_asymmetry(panel: np.ndarray) -> dict[str, float]:
    """Mean absolute differences from left/right and top/bottom reflections."""
    if panel.ndim != 2 or panel.size == 0 or not np.isfinite(panel).all():
        raise ValueError("The panel must be a non-empty, finite grayscale array")
    if panel.min() < 0 or panel.max() > 1:
        raise ValueError("Panel intensities must be normalized to [0, 1]")
    return {
        "horizontal_asymmetry": float(np.abs(panel - panel[:, ::-1]).mean()),
        "vertical_asymmetry": float(np.abs(panel - panel[::-1, :]).mean()),
    }


def extract_image_features(path: Path) -> dict[str, float]:
    """Read one image and return exactly two numerical measurements in [0, 1].

    Zero means perfect reflection symmetry, not a verified normal component.
    Marks, missing elements, lighting and imperfect alignment can affect both values.
    No reference image is learned, and no other image or label is consulted.
    """
    return measure_asymmetry(prepare_panel(path))


# B3 - calculate the same two features for every training record, without dropping failures.
def build_feature_table(train: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Preserve image identity and metadata; extract features from pixels only."""
    if train["image_id"].isna().any() or train["image_id"].duplicated().any():
        raise ValueError("Feature extraction requires non-missing, unique image IDs")
    records = []
    for row in train.itertuples(index=False):
        try:
            records.append(extract_image_features(data_dir / row.file))
        except (OSError, ValueError, cv2.error) as error:
            raise ValueError(f"Feature extraction failed for {row.image_id}: {error}") from error
    features = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    if len(features) != len(train) or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must be finite and available for every training record")
    # Inspector type is annotation information, not an image feature for prediction.
    metadata = train.drop(columns=["inspector_type"], errors="ignore").reset_index(drop=True)
    return pd.concat([metadata, features], axis=1)


# B3 - one feature comparison: horizontal asymmetry by inspector label.
def compare_asymmetry(features: pd.DataFrame, output_dir: Path, show: bool = True) -> None:
    """Show each observation alongside robust summaries, without causal claims."""
    summary = features.groupby("inspector_label")["horizontal_asymmetry"].agg(
        image_count="size",
        mean="mean",
        median="median",
        q1=lambda values: values.quantile(0.25),
        q3=lambda values: values.quantile(0.75),
        minimum="min",
        maximum="max",
    ).reindex([0, 1])
    summary.to_csv(output_dir / "asymmetry_by_label.csv")
    print("\nPart B3 - Left/right asymmetry across all labeled training images")
    print(f"Both features were extracted successfully for all {len(features)} training images.")
    print("We compare horizontal asymmetry by inspector label; Q1 and Q3 bound the middle 50% of images.")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    groups = [features.loc[features["inspector_label"] == label, "horizontal_asymmetry"].to_numpy()
              for label in [0, 1]]
    colors = ["#087F83", "#BD7148"]
    boxes = ax.boxplot(groups, positions=[1, 2], widths=0.4, patch_artist=True,
                       showfliers=False, medianprops={"color": "#172B3A", "linewidth": 2})
    jitter_rng = np.random.default_rng(RANDOM_SEED)
    for position, values, color, box in zip([1, 2], groups, colors, boxes["boxes"]):
        box.set_facecolor(color)
        box.set_alpha(0.2)
        # Jitter moves display positions only; all feature values remain unchanged.
        x = position + jitter_rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(x, values, s=18, alpha=0.6, color=color, edgecolors="none", zorder=3)
    ax.set_xticks([1, 2], [f"Labeled normal\n(n={len(groups[0])})",
                           f"Labeled defective\n(n={len(groups[1])})"])
    ax.set_ylim(0, min(1, max(0.01, float(features["horizontal_asymmetry"].max()) * 1.15)))
    ax.set_ylabel("Left/right mean absolute difference (normalized units)")
    ax.set_title("Horizontal asymmetry by inspector label", pad=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.1, 0.025, "Each dot is one training image. Boxes show the middle 50%; dark lines show medians.",
             fontsize=9, color="#52616B")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "asymmetry_by_label.png", dpi=180)

    normal, defective = summary.loc[0], summary.loc[1]
    iqr_overlap = max(normal["q1"], defective["q1"]) <= min(normal["q3"], defective["q3"])
    overlap_description = (
        "The middle 50% asymmetry intervals overlap between the two label groups. "
        if iqr_overlap else "The middle 50% asymmetry intervals do not overlap between the two label groups. "
    )
    observation = (
        f"Median horizontal asymmetry is {normal['median']:.4f} for normal-labeled images "
        f"and {defective['median']:.4f} for defective-labeled images. "
        + overlap_description
        + "This is a descriptive comparison, not a validation result or proof of product defects; "
        "uneven lighting, normal design details and alignment errors can also affect symmetry."
    )
    (output_dir / "asymmetry_comparison.txt").write_text(observation + "\n", encoding="utf-8")
    print(f"\nWhat the comparison shows\n{observation}")
    print(f"Saved comparison: {output_dir / 'asymmetry_by_label.png'}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    options = parse_run_options("part_b", "Part B: visual examples and simple image features.")
    if options.no_show:
        plt.switch_backend("Agg")
    data_dir, output_dir = options.data_dir, options.output_dir

    # Shared step - reuse the label validation already reviewed in Part A.
    train = load_training_data(data_dir / "train.csv")

    # B1 - display exactly six examples and save their metadata for traceability.
    examples = select_examples(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "file", "inspector_label", "camera", "exposure_setting", "batch"]
    examples[columns].to_csv(output_dir / "selected_examples.csv", index=False)
    print("\nPart B1 - A first look at the training images")
    print("We selected three images labeled normal and three labeled defective for a balanced visual comparison.")
    print("Selection was random within each label group, with seed=42 so the same examples can be reproduced.")
    print("This display does not represent the class proportions in the full dataset.\n")
    display_table = examples[["image_id", "inspector_label", "camera", "exposure_setting", "batch"]].copy()
    display_table["inspector_label"] = display_table["inspector_label"].map({0: "Normal (0)", 1: "Defective (1)"})
    display_table.columns = ["Image", "Inspector label", "Camera", "Exposure", "Batch"]
    print(display_table.to_string(index=False))
    observation_path = output_dir / "visual_observation.txt"
    observation_path.write_text(VISUAL_OBSERVATION + "\n", encoding="utf-8")
    print(f"\nWhat we noticed\n{VISUAL_OBSERVATION}")
    print("\nThe figure, selected image details and visual observation have been saved in:")
    print(output_dir)
    if options.no_show:
        print("Figure display is disabled; the image is saved for review.")
    else:
        print("The figure will now open. Close its window to finish this step.")
    display_examples(examples, data_dir, output_dir, show=not options.no_show)

    # B2 - demonstrate the feature function on one already selected training image.
    # Full-dataset extraction and class comparison belong to B3, not this step.
    example = examples.iloc[0]
    features = extract_image_features(data_dir / example["file"])
    print(f"\nPart B2 - Two measurements from {example['image_id']}")
    print("We align the bright panel, crop its border and normalize each image independently.")
    print(f"Left/right asymmetry: {features['horizontal_asymmetry']:.4f}.")
    print(f"Top/bottom asymmetry: {features['vertical_asymmetry']:.4f}.")
    print("Both values lie between 0 and 1; larger values mean less reflection symmetry.")
    print("Neither measurement alone establishes whether the component is defective.")
    pd.DataFrame([{"image_id": example["image_id"], **features}]).to_csv(
        output_dir / "feature_example.csv", index=False,
    )

    # B3 - extract both features for all training images and compare one feature by label.
    feature_table = build_feature_table(train, data_dir)
    feature_table.to_csv(output_dir / "train_image_features.csv", index=False)
    compare_asymmetry(feature_table, output_dir, show=not options.no_show)


if __name__ == "__main__":
    main()
