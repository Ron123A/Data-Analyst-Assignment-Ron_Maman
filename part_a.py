"""Part A: explore inspector labels in the supplied training metadata.

Run from any directory. Source data is read without modification.
Implements A1 (label distribution), A2 (camera comparison), and A3 (one chart and insight).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_options import parse_run_options


# Setup - analysis labels are fixed; runtime paths are resolved in main().
LABEL_NAMES = {0: "Labeled normal", 1: "Labeled defective"}


# Shared step - load training metadata and validate inspector labels.
def load_training_data(path: Path) -> pd.DataFrame:
    """Load metadata and reject missing or unexpected inspector labels."""
    frame = pd.read_csv(path)
    required_columns = {"image_id", "inspector_label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Training metadata contains no records")

    labels = frame["inspector_label"]
    invalid = labels.isna() | ~labels.isin(LABEL_NAMES)
    if invalid.any():
        bad_values = labels.loc[invalid].unique().tolist()
        raise ValueError(f"Missing or invalid inspector labels: {bad_values}")
    return frame


# A1 - calculate the overall inspector-label distribution.
def summarize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Count both label classes using all training records as denominator."""
    counts = frame["inspector_label"].value_counts().reindex([0, 1], fill_value=0)
    summary = pd.DataFrame({
        "inspector_label": [0, 1],
        "label_description": [LABEL_NAMES[0], LABEL_NAMES[1]],
        "record_count": counts.to_numpy(),
        "percentage": counts.to_numpy() / len(frame) * 100,
    })
    if summary["record_count"].sum() != len(frame):
        raise ValueError("Label counts do not reconcile with the record count")
    return summary


# A2 - calculate inspector-label counts and percentages for each camera.
def summarize_cameras(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare inspector-label rates, retaining rows with missing camera IDs."""
    if "camera" not in frame.columns:
        raise ValueError("Missing required column: camera")

    # Labels are validated as binary before this function is called.
    # Their sum therefore counts rows labeled defective within each camera.
    summary = frame.groupby("camera", dropna=False).agg(
        image_count=("inspector_label", "size"),
        labeled_defective_count=("inspector_label", "sum"),
    ).reset_index()
    summary["labeled_normal_count"] = (
        summary["image_count"] - summary["labeled_defective_count"]
    )
    summary["labeled_defective_percentage"] = (
        summary["labeled_defective_count"] / summary["image_count"] * 100
    )
    if summary["image_count"].sum() != len(frame):
        raise ValueError("Camera counts do not reconcile with the record count")
    if summary["labeled_defective_count"].sum() != frame["inspector_label"].sum():
        raise ValueError("Camera defective-label counts do not reconcile with the total")
    return summary[[
        "camera", "image_count", "labeled_normal_count",
        "labeled_defective_count", "labeled_defective_percentage",
    ]]


# A3 - compare batches to identify a group that warrants closer review.
def plot_batch_labels(frame: pd.DataFrame, output_dir: Path, show: bool = True) -> str:
    """Save and display one descriptive chart; do not infer a causal batch effect."""
    required = {"batch", "inspector_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing chart columns: {sorted(required.difference(frame.columns))}")

    # Retain missing metadata as an explicit category rather than dropping rows.
    groups = frame.assign(batch=frame["batch"].fillna("Missing batch")).groupby("batch").agg(
        image_count=("inspector_label", "size"),
        labeled_defective_count=("inspector_label", "sum"),
    ).reset_index()
    groups["labeled_defective_percentage"] = (
        groups["labeled_defective_count"] / groups["image_count"] * 100
    )
    if groups["image_count"].sum() != len(frame):
        raise ValueError("Chart group counts do not reconcile with training records")
    highest = groups.loc[groups["labeled_defective_percentage"].idxmax()]
    lowest = groups.loc[groups["labeled_defective_percentage"].idxmin()]
    gap = highest["labeled_defective_percentage"] - lowest["labeled_defective_percentage"]
    overall_rate = frame["inspector_label"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#087F83" if batch == highest["batch"] else "#8EA9BB" for batch in groups["batch"]]
    bars = ax.bar(groups["batch"], groups["labeled_defective_percentage"], color=colors, width=0.6)
    for bar, (_, record) in zip(bars, groups.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{record['labeled_defective_percentage']:.1f}%\n"
                f"{int(record['labeled_defective_count'])} / {int(record['image_count'])} images",
                ha="center", va="bottom", fontsize=10, color="#172B3A")
    ax.axhline(overall_rate, color="#52616B", linestyle="--", linewidth=1,
               label=f"Overall labeled-defective rate: {overall_rate:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Production batch")
    ax.set_ylabel("Images labeled defective (%)")
    ax.set_title("Inspector-label rates by production batch", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.1, 0.025, "Training data only. Labels are not independently verified defects.",
             fontsize=9, color="#52616B")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "batch_label_rates.png", dpi=180)
    groups.to_csv(output_dir / "batch_label_summary.csv", index=False)

    insight = (
        f"{highest['batch']} has the highest observed defect-label rate "
        f"({int(highest['labeled_defective_count'])}/{int(highest['image_count'])}, "
        f"{highest['labeled_defective_percentage']:.1f}%), compared with "
        f"{int(lowest['labeled_defective_count'])}/{int(lowest['image_count'])} "
        f"({lowest['labeled_defective_percentage']:.1f}%) in {lowest['batch']} "
        f"({gap:.1f} percentage points higher), making it a priority for reviewing images "
        "and capture conditions rather than evidence of a causal production problem or a sustained rise."
    )
    (output_dir / "part_a_insight.txt").write_text(insight + "\n", encoding="utf-8")
    # Show the saved figure during an interactive run; close its window to continue.
    if show:
        plt.show()
    plt.close(fig)
    return insight


def main() -> None:
    options = parse_run_options("part_a", "Part A: inspector labels, cameras and production batches.")
    if options.no_show:
        plt.switch_backend("Agg")
    data_dir, output_dir = options.data_dir, options.output_dir

    # Shared step - read and validate the source data once.
    train = load_training_data(data_dir / "train.csv")

    # A1 - display and save record counts and the overall label distribution.
    summary = summarize_labels(train)

    print("Part A1 - Training records and inspector labels")
    print(f"Training records: {len(train)}")
    print(f"Unique image IDs: {train['image_id'].nunique(dropna=True)}")
    print(f"Missing image IDs: {train['image_id'].isna().sum()}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "label_distribution.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved label distribution: {summary_path}")
    print("These counts describe inspector labels, not independently verified defects.")

    # A2 - display and save the camera comparison, including group sizes.
    camera_summary = summarize_cameras(train)
    print("\nPart A2 - Inspector-label rates by camera")
    print(f"Records with missing camera: {train['camera'].isna().sum()}")
    print(camera_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    camera_path = output_dir / "camera_label_summary.csv"
    camera_summary.to_csv(camera_path, index=False)
    print(f"\nSaved camera summary: {camera_path}")

    # A3 - save one chart and one evidence-based descriptive insight.
    insight = plot_batch_labels(train, output_dir, show=not options.no_show)
    print("\nPart A3 - One chart and one insight")
    print(insight)
    print(f"Saved chart: {output_dir / 'batch_label_rates.png'}")


if __name__ == "__main__":
    main()
