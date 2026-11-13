# Face recognition and authentication

Maurice's workspace for the Formative 2 face-recognition component, on branch
`feature/face`. It includes image analysis, filters, training-only augmentation,
pretrained YuNet/SFace features, classifier training, validation, held-out testing
and model export.

Start with [02_face_recognition.ipynb](notebooks/02_face_recognition.ipynb).
See the [setup and workflow guide](docs/face_recognition.md) for local Anaconda,
Google Colab and Kaggle configuration.

## Setup with Anaconda

```bash
conda env create -f environment-face.yml
conda activate multimodal-face
python -m ipykernel install --user --name multimodal-face --display-name "Python (multimodal-face)"
jupyter lab
```

Select the `Python (multimodal-face)` kernel. For an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Workspace

```text
.
├── README.md
├── requirements.txt
├── environment-face.yml
├── docs/face_recognition.md
├── notebooks/02_face_recognition.ipynb
├── face_auth/
│   ├── __init__.py
│   ├── embeddings.py                # Detection, alignment, pretrained features
│   ├── pipeline.py                  # Splits, augmentation, training, rejection
│   └── __main__.py                  # Face authentication CLI
├── feature_extraction.py            # Reusable face extractor
├── data/
│   ├── raw/images/<user_id>/
│   └── processed/                   # Generated manifest and image features
├── models/                          # Metadata, saved classifier and weights
├── reports/face/                    # Generated plots, metrics and executed notebook
└── tests/test_face_pipeline.py
```

Four members are enrolled, each with three photographs (neutral, smiling,
surprised): Kudakwashe, Larissa, Maurice and Patrick. A person is enrolled only
with at least three independent valid photographs; anyone with fewer is used as
an unknown person in testing. `data/raw/images/unauthorized/` holds 14 photos of
non-members that must be rejected.

## Training and evaluation

Run the notebook from top to bottom, locally, on Google Colab or on Kaggle. It
works from the project folder, so every saved path is relative. Photos are resized
to at most 1024 px, the original photos are split (neutral → train, smiling →
validation, surprised → test) **before** augmentation, and each photo then gets 12
augmented copies that stay in its split. Faces are encoded with the pretrained
SFace 2021dec model (128 values) through `extract_face_features()` in
`feature_extraction.py`, which the command-line tool also calls.

Latest run: validation and test accuracy 100 % (test log loss 0.068 vs 1.386 for a
dummy baseline); rejection threshold 0.417; all 14 unauthorized photos (and their
168 augmented copies) denied; 0 of 12 stranger attempts accepted when leaving one
member out; all 21 command-line checks passed. The counts are small, so treat them
as evidence the method works rather than a precise error rate.

Generated outputs:

- `data/processed/resized_images/` (photos resized to ≤ 1024 px)
- `data/processed/face_manifest.csv`
- `data/processed/image_features.csv` (`user_id`, `source_id`, `split`, `augmentation` + 128 features)
- `models/face_model.joblib`
- `models/model_metadata.json` (`face.threshold`, `face.pretrained_model`, `face.feature_columns`)
- `models/pretrained/` containing the two ONNX weights
- `reports/face/` containing audits, plots, metrics, test attempts and the executed notebook

The trained model and generated biometric outputs are available locally and
ignored by Git. Rerun the notebook to recreate them in another environment.

## Authenticate a face

```bash
python -m face_auth --image /path/to/new-photo.jpg
python -m pytest -q
```

Exit codes: `0` verified, `1` denied, `2` setup error. Missing, multiple, small or
unverified faces are denied. Only load trusted model files. This classroom
workflow has no liveness or replay defense. Obtain participant consent before
sharing photographs, embeddings, models or executed notebook outputs.

Notebook section **5.3** saves **21 processed PNGs per enrolled person** from
three originals (original, rotation, flip, grayscale, CLAHE, gamma and median blur per source). Files appear
in `data/processed/images/<user_id>/` when the cell runs, with descriptive names
and a provenance CSV. People with fewer than three originals are listed as
skipped. The exported folder stays separate from model training to preserve
original-source train/validation/test isolation.
