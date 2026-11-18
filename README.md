# Junior Data Analyst Assignment

This submission covers Parts A-E. The Python files reproduce the analysis, and `outputs/` contains the tables, figures, validation results and test predictions.

Required predictions: [`outputs/part_c/predictions.csv`](outputs/part_c/predictions.csv).

## Part E - Conclusions

The training set contains 168 images: 112 labeled normal and 56 labeled defective. CAM-B has a slightly higher defect-label rate than CAM-A, 35.6% of 73 images versus 31.6% of 95. The larger descriptive difference is between batches: 42.9% in BATCH-C and 26.2% in BATCH-B, with 42 images in each. These figures identify groups worth reviewing; they do not establish whether the cause is manufacturing quality, capture conditions or labeling. They also do not establish a sustained rise in defects.

The component has a repeated layout, so I used two measurements of reflection asymmetry, one comparing left with right and the other comparing top with bottom. Each image is aligned and normalized before measurement. A regularized logistic regression uses these two numbers, with scaling and fitting restricted to the 126 development images. The 42 validation images are excluded from fitting, and the decision threshold is 0.5.

On validation, the model detected 5 of the 14 images labeled defective, missed 9 and raised 6 false alarms among 28 normal-labeled images. Precision is 45.45% and recall is 35.71%. ROC-AUC is 0.607, indicating only modest ranking ability. These results are not strong enough to support automatic acceptance or rejection of components.

The review of IMG_0062 illustrates a false alarm: the inspector labeled it normal, but its score of 0.5135 is just above the threshold. The lower lettering has no matching text above the upper row and contributes to legitimate top/bottom asymmetry. Lighting and alignment may also contribute; this is a possible explanation, not a confirmed cause. Conversely, small or symmetric defects can be missed, and cropping can remove edge damage.

The evaluation measures agreement with imperfect inspector labels. It uses a small image-level split that can share cameras and batches, and Parts A-B describe all labeled records. Feature alternatives were considered during development after the validation results had already been inspected, so the reported metrics are a preliminary estimate rather than an independent final test. I would next review disagreements with the inspector and check performance on separately collected batches and capture conditions before considering operational use. The 72 test predictions are model outputs, not verified outcomes; test data was used only for feature extraction and inference.

I used AI to help draft the code, interpret the results and refine the explanations. The outputs were checked by running the full pipeline, inspecting images and independently verifying feature calculations, validation metrics and the prediction file format.

## Setup

Use a Python environment with the packages in `requirements.txt`:

```text
python -m pip install -r requirements.txt
```

The code has been tested with Python 3.14.2 and scikit-learn 1.9.1. Use the same environment to install packages and run the scripts. The scikit-learn version is pinned because the saved joblib model was created with that version. Run Part C before Part D to recreate the fitted model in the installed environment; loading a saved scikit-learn model across versions is not supported by this submission.

## Run

Extract the ZIP, open a terminal inside `Ron_Maman_Assignment/` and run the following commands in order. The `data/` folder contains unchanged copies of the supplied CSV files and all 240 images, so no separate dataset download is needed:

```text
python part_a.py --data-dir data --no-show
python part_b.py --data-dir data --no-show
python part_c.py --data-dir data --no-show
python part_d.py --data-dir data --no-show
```

To use another copy of the original dataset, replace `data` with its folder path. That folder must contain `train.csv`, `test.csv` and `images/`. Image paths in the CSV are resolved relative to it.

The scripts can also be called by their full paths from any working directory. Absolute data paths avoid ambiguity; relative command-line paths are relative to the terminal's current directory.

If `--data-dir` is omitted, the code looks beside the scripts, in their parent folder, and in the terminal's current directory. It continues only if these locations identify exactly one dataset; otherwise it asks for an explicit path. For the packaged layout, use `--data-dir data` as shown above.

The commands above save figures without opening windows. To display them as well, omit `--no-show`. Close each figure window to continue. For example:

```text
python part_a.py --data-dir data
python part_b.py --data-dir data
```

## Outputs

Results are saved under `outputs/part_a/` through `outputs/part_d/` beside the scripts. To choose another output root:

```text
python part_a.py --data-dir "path/to/dataset" --output-dir "path/to/results" --no-show
```

This example writes Part A results to `path/to/results/part_a/`. Input files are read without modification. Run `python part_a.py --help` or `python part_b.py --help` for the available options.

The selected Part A chart is `batch_label_rates.png`; Part B1 saves `six_training_examples.png`, the six selected records and the written visual observation.

## The two image features

The component has a repeated, approximately symmetric layout. Missing elements or extra marks may disturb this layout. Part B therefore extracts exactly two numbers from each image:

- `horizontal_asymmetry`: the mean absolute difference between the prepared panel and its left/right reflection.
- `vertical_asymmetry`: the mean absolute difference between the prepared panel and its top/bottom reflection.

For each pixel, take the absolute intensity difference from its reflected counterpart, then average over the panel. Both values lie in [0, 1]. Zero means perfect reflection symmetry; it does not establish that the component is normal. Real design details, such as the lettering near the lower edge, need not be symmetric.

Before measurement, the image is converted to grayscale. Otsu thresholding and the largest external contour identify the bright panel against its dark background. A minimum-area rectangle is aligned to a 128 by 128 image using linear interpolation. Six pixels are cropped from every edge, leaving 116 by 116 pixels. The panel's own 5th and 95th intensity percentiles define a linear normalization, clipped to [0, 1]. This reduces sensitivity to overall exposure; it does not remove uneven lighting. The rectangle corrects in-plane rotation, not arbitrary perspective.

These preprocessing settings are fixed implementation choices, not universal standards or claimed optima. They use only the current image, with no labels, learned reference image or statistics from other records. An unreadable image, foreground smaller than 10% of the image, or percentile range below one grayscale unit stops extraction with a descriptive error. Failed records are not silently discarded.

The approach assumes a bright rectangular panel on a dark background and modest rotation. Symmetric defects may be missed; cropping can remove edge damage; noise and inaccurate alignment can create false alarms. A low score is not a product-quality guarantee.

Part B saves `train_image_features.csv` with both features for every training image, and one required feature comparison, `asymmetry_by_label.png`, showing horizontal asymmetry by inspector label. Its summary table and short interpretation are saved alongside it. Inspector type is excluded from the feature table because it is annotation information, not a pixel-derived measurement. No test images are used in Part B.

Part C uses a fixed, stratified 75/25 development/validation split (seed 42), after checking for exact decoded-image duplicates. It saves the split membership, feature tables, split summary, run configuration and `development_pipeline.joblib` under `outputs/part_c/`.

A scikit-learn Pipeline combines StandardScaler with LogisticRegression, fitting both on development data only and selecting only `horizontal_asymmetry` and `vertical_asymmetry` as inputs. Camera, batch, image ID and inspector information are not model inputs. The L2 penalty and score threshold of 0.5 are fixed. The inverse regularization parameter is set to `C = 1 / (development_record_count * 0.01)`. The model score is not a calibrated defect probability.

The validation subset is held out from model fitting, but Parts A-B explored all labeled records. No parameters or thresholds were selected using test data. The image-level split may share batches and cameras across subsets; it does not establish generalization to unseen batches or cameras. The duplicate check detects identical decoded pixels, not near-duplicates. No test data is used in C1-C2.

Part C3 requires the supplied `test.csv` with 72 unique image IDs. It extracts the same two features and reloads the saved development pipeline for inference, without fitting or tuning on test data. The required submission file is `outputs/part_c/predictions.csv`, with exactly `image_id,predicted_defect,score` and one row per test image in its original order. The code checks the written CSV after saving it. `test_image_features.csv` is retained for traceability; there are no test labels or test accuracy estimates.

Run Part C before Part D. Part D reads `part_c/` under the same output root and writes validation predictions, a confusion matrix (PNG and CSV) and metrics (JSON and CSV) to `part_d/`. Use the same `--output-dir` root for both scripts if changing the default. It performs no fitting, threshold tuning or test-data access. Precision and recall concern the positive class (inspector label 1). ROC-AUC is computed directly from the validation scores, independently of the decision threshold, and saved as `roc_auc`. Undefined metrics are explained in the terminal and represented as `null` in JSON or a blank CSV cell, with an accompanying defined/undefined flag; ROC-AUC requires both classes.

Part D2 selects the misclassified validation image closest to the fixed threshold, breaking ties by image ID. If there are no mistakes, it selects the closest validation image instead. It saves `validation_case.csv`, the original and prepared image in `validation_case_review.png`, and a possible explanation in `validation_case_review.txt`. The explanation concerns disagreement with the inspector, not an independently verified defect diagnosis.

## Submission contents

Keep the files and folders in their existing structure. The original inputs are included under `data/` and are read without modification. No notebook, slide deck or separate PDF is needed.

| File | Purpose |
| --- | --- |
| `part_a.py` | A1-A3: label counts, camera comparison, one batch chart and insight. |
| `part_b.py` | B1-B3: six example images, the two image features and one feature comparison. |
| `part_c.py` | C1-C3: split, development-only fitting and test predictions. |
| `part_d.py` | D1-D2: validation metrics, confusion matrix and one error review. |
| `run_options.py` | Shared input/output paths and figure-display options; required by A-D. |
| `requirements.txt` | Python dependencies. |
| `README.md` | Run instructions, method, Part E conclusions, and this file guide. |
| `data/train.csv` | Original metadata and inspector labels for the 168 training images. |
| `data/test.csv` | Original metadata for the 72 unlabeled test images. |
| `data/images/` | All 240 original images, with their supplied filenames and bytes unchanged. |

All paths below are relative to `outputs/`.

| Output file | Purpose |
| --- | --- |
| `part_a/label_distribution.csv` | Counts and percentages of the two inspector labels. |
| `part_a/camera_label_summary.csv` | Image counts and defect-label rates for each camera. |
| `part_a/batch_label_rates.png` | The selected A3 chart. |
| `part_a/batch_label_summary.csv` | Values underlying the batch chart. |
| `part_a/part_a_insight.txt` | The written A3 insight. |
| `part_b/six_training_examples.png` | The six B1 examples, shown in original colors. |
| `part_b/selected_examples.csv` | Identities and metadata of those six images. |
| `part_b/visual_observation.txt` | The short visual observation for B1. |
| `part_b/feature_example.csv` | The two measurements for one B2 example. |
| `part_b/train_image_features.csv` | Both features for all 168 training images. |
| `part_b/asymmetry_by_label.png` | The required B3 comparison of one feature. |
| `part_b/asymmetry_by_label.csv` | Numerical summaries underlying that comparison. |
| `part_b/asymmetry_comparison.txt` | Interpretation of the B3 comparison. |
| `part_c/split_assignments.csv` | Development/validation membership for every training image. |
| `part_c/split_summary.csv` | Subset sizes and label counts. |
| `part_c/development_features.csv` | The 126 development records and their measurements. |
| `part_c/validation_features.csv` | The 42 validation records and their measurements. |
| `part_c/development_pipeline.joblib` | Saved fitted scaler and classifier, required when running D without rerunning C. |
| `part_c/baseline_specification.json` | Model settings, input features and decision threshold. |
| `part_c/run_config.json` | Split and image-preprocessing settings, software version and evaluation scope. |
| `part_c/test_image_features.csv` | The two measurements for each test image, retained for traceability. |
| `part_c/predictions.csv` | Required deliverable: 72 test predictions with exactly three columns. |
| `part_d/validation_predictions.csv` | Validation labels, measurements, scores and predicted labels. |
| `part_d/validation_confusion_matrix.png` | The required D1 confusion matrix. |
| `part_d/validation_confusion_matrix.csv` | The same confusion-matrix counts in tabular form. |
| `part_d/validation_metrics.csv` | Precision, recall, ROC-AUC, counts and threshold in tabular form. |
| `part_d/validation_metrics.json` | The same metrics in structured form. |
| `part_d/validation_case.csv` | The single validation record selected for D2. |
| `part_d/validation_case_review.png` | Original and prepared views of that image. |
| `part_d/validation_case_review.txt` | The possible explanation and its limitations. |

