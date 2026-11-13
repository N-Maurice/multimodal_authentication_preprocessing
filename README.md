# Multimodal authentication and product recommendation

Formative 2 team project: verify a face, verify the same person's voice, then
predict a product category from that customer's tabular features.

This repository contains the agreed structure, starter notebooks and a CLI
scaffold. Models, feature extraction, evaluation results and the final PDF still
need to be developed. No trained artifacts or measured results are included.

## Team ownership

| Member | Notebook | Responsibilities | Saved output |
|---|---|---|---|
| Patrick | 01_product_recommendation.ipynb | Merge, engineer features, train and evaluate product prediction | product_model.joblib |
| Maurice | 02_face_recognition.ipynb | Image collection, augmentation, pretrained features, identity classification and rejection | face_model.joblib |
| Kudakwashe | 03_voice_verification.ipynb | Audio collection, augmentation, speaker features and verification | voice_model.joblib |
| Larissa | 04_system_integration.ipynb and main.py | Registry, authentication sequence, integration tests, CLI and video | Working application |

## Face authentication (implemented)

The face part is complete: `notebooks/02_face_recognition.ipynb`, the `face_auth/`
package and `extract_face_features()` in `feature_extraction.py`. It uses the
pretrained OpenCV Zoo SFace encoder (`face_recognition_sface_2021dec`, 128-value
embedding) with the YuNet detector (`face_detection_yunet_2023mar`); both ONNX files
are downloaded to `models/pretrained/` by the notebook and not committed. All four
members have neutral, smiling and surprised photos, and `data/raw/images/unauthorized/`
holds 14 non-member photos used only to test rejection. The rejection threshold is
stored as `face.threshold` in `models/model_metadata.json`. Workflow, results and
Colab/Kaggle instructions: [docs/face_recognition.md](docs/face_recognition.md).
Face tests: `tests/test_face_pipeline.py`; face-only environment: `environment-face.yml`.

## Setup

Use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Pretrained face/speaker encoders are not selected yet. Maurice and Kudakwashe
must agree on suitable encoders, add their dependencies and document model
licenses/downloads before implementing the shared extractors.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles - customer_social_profiles.csv
│   │   ├── customer_transactions - customer_transactions.csv
│   │   ├── images/<user_id>/
│   │   └── audio/<user_id>/
│   └── processed/
│       └── user_registry.csv
├── notebooks/
│   ├── 01_product_recommendation.ipynb
│   ├── 02_face_recognition.ipynb
│   ├── 03_voice_verification.ipynb
│   └── 04_system_integration.ipynb
├── models/model_metadata.json
├── feature_extraction.py
├── main.py
├── reports/
│   ├── final_report.md
│   └── contributions.md
└── tests/test_system.py
```

The existing raw filenames are preserved. Training must generate
`data/processed/merged_dataset.csv`, `image_features.csv`, `audio_features.csv`
and `customer_inference.csv`, plus all three `.joblib` models. Export the finished
report as `reports/final_report.pdf`. These outputs are deliberately not empty
placeholder files that could be mistaken for completed work.

## Data currently available

The social CSV has 155 rows and the transaction CSV has 150 rows. Their key names
differ (`customer_id_new` and `customer_id_legacy`); Patrick must validate the
mapping before merging.

At initial inventory, Maurice and Larissa each have three facial images;
Kudakwashe has one, and Patrick has none. Maurice and Larissa each have two audio
recordings; Kudakwashe and Patrick have none. Add the missing samples and separate
unauthorized attempts. More independent captures are needed for credible held-out
evaluation. OGG decoding may require a compatible system audio decoder.

## Shared integration contracts

- Use the same lowercase `user_id` across biometric folders, model class labels,
  feature CSVs and the registry. Registry columns are `user_id,display_name,customer_id`.
  Assign real tabular customer IDs explicitly for this classroom simulation.
- Each biometric artifact is a fitted estimator or Pipeline with `classes_` and
  `predict_proba`. Shared extractors in `feature_extraction.py` return a single
  two-dimensional feature row and must be identical in training and inference.
- Feature CSVs retain `user_id,source_id,split,augmentation` alongside numeric
  features. Never feed identity labels or provenance columns into the classifier.
- Product output is a fitted preprocessing/classifier Pipeline. Record its input
  columns in `models/model_metadata.json`. `customer_inference.csv` contains
  exactly one pre-purchase feature row per `customer_id` and no target column.
- Save calibrated face/voice thresholds and encoder details in metadata. Set
  `status` to `trained` after completing and validating the artifacts.
- Both identities must be enrolled and match. A voice belonging to another
  enrolled person must fail. Prediction runs only after both checkpoints pass.

Do not merge face/audio sample rows directly into transaction rows: their
one-to-many relationships can multiply observations and leak identity. Link
modalities through the registry for the sequential system; if experimenting with
a fused model, aggregate features per user within training splits first.

## Evaluation and demonstration

Split original captures before augmentation; keep each original and all its
variants in one partition. Use validation data to select thresholds, then report
held-out accuracy, macro F1 and log loss, plus false acceptance/rejection rates
and unknown-person tests. With very few original samples, report limitations
explicitly. Keep recommendation splits grouped by customer or time as appropriate.

After completing all notebooks and extractors:

```bash
python main.py --face data/raw/images/maurice/Maurice_Neutral.jpeg \
  --voice data/raw/audio/maurice/Maurice_Approved.ogg
python -m pytest -q
```

The example paths illustrate invocation; use held-out samples for evaluation.
The CLI currently reports a setup error until extractors and artifacts exist.
Exit codes: `0` success, `1` authentication denied, `2` missing/invalid setup.
This is a classroom simulation without liveness or replay protection. Classifier
confidence alone does not establish biometric security. Only load trusted joblib
artifacts. Obtain participant consent before committing or publishing biometrics.

Complete the report, contributions, repository link and demonstration video link
before submission.
