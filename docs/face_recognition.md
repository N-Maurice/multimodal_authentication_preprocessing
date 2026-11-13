# Face authentication workflow

Owner: Maurice. Development branch: `feature/face`.

Start with [the complete notebook](../notebooks/02_face_recognition.ipynb).
Run every cell in order; configuration is in cell 1.1. The notebook performs real
training, evaluation and export, rather than supplying placeholder metrics.

## Navigation

| Location | Purpose |
|---|---|
| `notebooks/02_face_recognition.ipynb` | Narrative, analysis, filters, training, validation, testing and export |
| `face_auth/embeddings.py` | YuNet detection, alignment, pretrained SFace features and explicit weight downloads |
| `face_auth/pipeline.py` | Data audit, safe splits, training augmentation, classifier, threshold calibration, leave-one-identity-out |
| `face_auth/preparation.py` | Export of 21 processed images per enrolled person |
| `face_auth/__main__.py` | Face-only inference CLI |
| `feature_extraction.py` | Reusable face extractor for training/inference parity |
| `requirements.txt` | Face-only pip environment |
| `environment-face.yml` | Anaconda environment |
| `tests/test_face_pipeline.py` | Data isolation, duplicates, rejection, calibration, export and serialization tests |

## Notebook sections

| Part | Section | Content |
|---|---|---|
| Setup | 1 | Environment detection, packages, project discovery (or clone); works from the project folder so all paths are relative |
| A. Data | 2 | 2.1 coverage (4 members × neutral/smiling/surprised); 2.2 resize to ≤ 1024 px; 2.3 quality and duplicates; 2.4 sample images of consented members |
| | 3 | Face detection (YuNet 2023mar) and alignment for all 12 photos, crop sharpness |
| | 4 | Split of the **original** photos before any augmentation: neutral → train, smiling → validation, surprised → test |
| B. Augmentation and features | 5 | 5.1 twelve augmentations on every photo, copies kept in their source split; 5.2 "still the same person?" check; 5.3 21 exported PNGs per member |
| | 6 | `extract_face_features` (SFace 2021dec, 128 values), `image_features.csv`, per-photo checklist |
| | 7 | PCA embedding map and validation-vs-member similarity |
| C. Model | 8 | Dummy baseline, `C` comparison and validation curve |
| | 9 | Rejection threshold on validation originals, four-panel validation figure |
| D. Results | 10 | Held-out test (originals + augmented copies); 10.1 unauthorized-image test; 10.2 similarity matrix |
| | 11 | 11.1 leave-one-member-out stranger check; 11.2 all six expression assignments |
| | 12 | Save `face_model.joblib`, update the `face` section of `model_metadata.json`, reload and extractor parity checks |
| | 13 | Command-line authentication checks |
| | 14 | Try your own photograph |
| E. Conclusions | 15 | Summary table, interpretation, limitations, next steps; 15.1 zip outputs on Colab/Kaggle |

Every section that produces results ends with a *What this shows* block computed
from that run.

## Local Anaconda

```bash
conda env create -f environment-face.yml
conda activate multimodal-face
python -m ipykernel install --user --name multimodal-face --display-name "Python (multimodal-face)"
jupyter lab
```

Select the `Python (multimodal-face)` kernel. For an existing environment, use
`python -m pip install -r requirements.txt` with that environment activated.
Locally the notebook never installs packages itself; it stops and lists what is
missing. No GPU or CUDA is needed. Do not install two OpenCV wheels in one
environment.

## Google Colab

Use the *Open in Colab* badge at the top of the notebook, or *File → Open
notebook → GitHub* and paste the repository URL. Run all cells. Section 1.3
clones branch `feature/face` (code and images) into `/content` automatically;
**the branch must be pushed to GitHub first**. Every required package is already
installed on Colab; section 1.2 verifies versions and installs anything missing.

To work from your own copy on Google Drive, mount Drive and set
`PROJECT_ROOT_OVERRIDE`. Files in `/content` are deleted when the runtime ends;
section 14.1 zips the outputs and starts a download.

## Kaggle

Import the notebook (*File → Import notebook → GitHub*) and turn on
*Settings → Internet*. Section 1.3 clones the repository into `/kaggle/working`
and the face models are downloaded on first run. Outputs, including
`face_outputs.zip`, appear in the *Output* tab after saving a version.

Offline alternative: attach the repository and the two ONNX files as private
datasets, set `PROJECT_ROOT_OVERRIDE` to the folder containing `face_auth/`,
`WEIGHTS_DIR_OVERRIDE` to the ONNX folder and `DOWNLOAD_WEIGHTS = False`. A
read-only project is detected automatically and outputs go to
`/kaggle/working/multimodal-face`.

## Data and evaluation policy

Use `data/raw/images/<user_id>/<filename>` with the expression in the filename
(`Name_Neutral`, `Name_Smiling`, `Name_Surprised`). Raw photos are never modified:
section 2.2 writes copies with the longest side at most 1024 px (Patrick's
3024-px photos and every other photo above 1024 px are reduced) to
`data/processed/resized_images/`, and every later step uses those copies.

The **original** photos are split first (`SPLIT_BY_EXPRESSION = True`: neutral →
train, smiling → validation, surprised → test; `False` uses a seeded random split).
Only then are augmented copies created; each copy inherits the split of its source
photo, and the code asserts that no source appears in two splits. Members without
all three expressions, and the `unauthorized/` or `unknown/` folders, are held out
entirely as unknown test people. `unauthorized/` currently holds 14 web/stock
portrait photos of non-members; they are tested (with their augmented copies) in
sections 10.1, 11.2 and 13 but not displayed unless `SHOW_UNAUTHORIZED_IMAGES = True`.
Section 11.1 also measures stranger rejection by leaving one member out at a time.
Only members listed in `CONSENTED_MEMBERS` are displayed in the notebook.

Twelve augmentations are applied to every photo (members and unauthorized): ±8° rotation, horizontal flip,
affine warp (5 % zoom-out, 0.08 shear, 3-px shift), grayscale, 10 % darker, 10 %
more contrast, gamma 0.8, CLAHE, Gaussian blur, 3×3 median blur and sharpening.
Section 5.2 records, for every copy, whether it still looks like the same person
(similarity to its original ≥ 0.363 and higher than to any other member).

Logistic regression `C` is selected on all validation rows from 0.01–100 (capped
to avoid over-confident probabilities). The rejection threshold (`face.threshold`)
is calibrated on the validation **original** photos only, so synthetic copies do
not pull it towards accepting more; when several thresholds score equally, the
middle of the tied range is used. The test set is untouched until the final
evaluation, and the saved model is not refit afterward.

## Generated outputs

All paths are relative to the project folder (the notebook works from there).

- `data/processed/resized_images/<user_id>/` and `resized_images.csv`: photos resized to ≤ 1024 px.
- `data/processed/face_manifest.csv`: every original photo, expression and split.
- `data/processed/image_features.csv`: exactly `user_id`, `source_id`, `split`,
  `augmentation`, then `embedding_000` … `embedding_127` (338 rows: 26 photos × 13 versions).
- `data/processed/images/<user_id>/`: 21 processed PNGs per enrolled member.
- `models/face_model.joblib`: `FaceAuthenticator` with `classes_` and `predict_proba`,
  member centroids and both thresholds.
- `models/pretrained/*.onnx`: the two pretrained models; downloaded explicitly.
- `models/model_metadata.json`: only the `face` section is written (voice, product
  and integration sections are preserved): `threshold`, `pretrained_model`,
  `feature_columns`, encoder/detector names and versions, embedding size, hashes,
  versions and evaluation.
- `reports/face/`: every figure and table, including `augmentation_identity_check.csv`,
  `cli_checks.csv` and `metrics.json`.

Generated biometric files and weights are ignored by Git. The notebook is saved
with its outputs, which include participants' photographs: share it only with
everyone's consent, or clear outputs first.

## Feature extractor

`feature_extraction.extract_face_features(path, weights_dir=None)` is the single
function used by the notebook check and the command-line tool:

| Property | Value |
|---|---|
| Encoder | SFace, OpenCV Zoo `face_recognition_sface_2021dec.onnx` |
| Detector | YuNet, OpenCV Zoo `face_detection_yunet_2023mar.onnx` |
| Embedding size | 128, L2-normalised |
| Returns | NumPy array of shape `(1, 128)` |
| Rejects (`FaceInputError`, a `ValueError`) | unreadable file, no face, several faces, face side < 30 px |

## Face inference

After running the notebook, from the repository root:

```bash
python -m face_auth --image /path/to/new-photo.jpg
# For cloud or relocated outputs:
python -m face_auth --image /path/to/new-photo.jpg \
  --model /path/to/output/models/face_model.joblib \
  --weights /path/to/onnx-directory
python -m pytest -q
```

Exit codes: 0 verified face, 1 denied input/identity, 2 setup failure. Missing,
multiple or too-small faces are rejected. A face must pass both the classifier
confidence gate and the cosine gate for the predicted identity.

`FaceAuthenticator.predict_proba` returns zero scores for rejected faces; those
rows intentionally do not sum to one. Use `classifier_` probabilities for log loss.
Only load trusted joblib files. This classroom workflow has no liveness or replay
defense.

## Latest verified run

26 photographs were processed: neutral, smiling and surprised for Kudakwashe,
Larissa, Maurice and Patrick, plus 14 unauthorized photos. 11 of the 12 member
photos were reduced to ≤ 1024 px (Patrick's from 3024 px), and every photo had
exactly one detectable face. 312 augmented copies were created; all 144 member
copies still match their own person. Selected `C = 100`; rejection threshold
`face.threshold` = 0.417; confidence threshold 0.5.

- Validation: accuracy 100 %, macro F1 1.00, log loss 0.140 (dummy baseline 1.386);
  4/4 originals and 50/52 rows authenticated (2 heavily filtered Kudakwashe copies denied).
- Test originals: accuracy 100 %, macro F1 1.00, log loss 0.068; 4/4 accepted as the
  right member. Augmented test copies: 98 % accepted, 0 % accepted as the wrong member.
- Unauthorized: 14/14 photos and 182/182 rows including augmented copies denied;
  highest stranger similarity 0.31 vs threshold 0.417 (lowest member test score 0.48).
- Leave-one-member-out: 0 of 12 stranger attempts accepted; 11 of 12 genuine accepted.
- All six expression assignments: 100 % of genuine test photos accepted; no hidden
  member or unauthorized row accepted.
- Command line: 21/21 checks (4 members verified; 14 unauthorized photos, a blank
  image, two faces and a tiny face denied).

The notebook ran end to end locally (Python 3.14). Before the unauthorized photos
were added, the same version also ran on a fresh copy in a Colab-like environment
(Python 3.12, OpenCV 4.10, scikit-learn 1.6, pandas 2.2, NumPy 2.0) with automatic
weight download, and offline on a read-only copy. Hosted Colab/Kaggle sessions were
not run from here; they only see the unauthorized photos once those are pushed.

## Processed images (section 5.3)

For each enrolled person, the first three resized photographs in filename order produce
seven versions each: original, +8° rotation, horizontal flip, grayscale, CLAHE,
gamma correction (0.8) and 3×3 median blur, giving twenty-one files. Filenames
contain user ID, source number, original stem, hash and transform; reruns reuse
the same names. `processed_image_manifest.csv` records each file's original and
split; `processed_image_skipped.csv` lists skipped people. These full-photo
exports are a deliverable only: training uses aligned crops with all twelve
variants and never reads the exported folder.
