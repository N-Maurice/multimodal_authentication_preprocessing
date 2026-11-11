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
├── requirements.txt                 # Includes requirements/face.txt
├── requirements/face.txt
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

All members' facial photographs remain because they are inputs to your identity
classifier and unknown-person tests. Currently Maurice and Larissa each have
three images; Kudakwashe has one and is withheld as an unknown identity.
Enroll an identity only after collecting at least three independent valid images.
Use `data/raw/images/unauthorized/` for additional unknown-person samples.

## Training and evaluation

Run the notebook from top to bottom. It audits image quality and duplicates,
freezes source-level train/validation/test partitions, augments training crops,
extracts 128-dimensional embeddings, compares classifiers on validation, selects
a rejection threshold, evaluates held-out originals, and saves the model.

The initial run identified two known test images correctly and rejected one
unknown image. That sample count is too small for reliable performance claims.
Collect more independent sessions and unknown people for stronger evaluation.

Generated outputs:

- `data/processed/face_manifest.csv`
- `data/processed/image_features.csv`
- `models/face_model.joblib`
- `models/model_metadata.json`
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

Notebook section **5.1** saves **12 processed PNGs per eligible person** from
three originals (original, rotation, flip and grayscale per source). Files appear
in `data/processed/images/<user_id>/` when the cell runs, with descriptive names
and a provenance CSV. People with fewer than three originals are listed as
skipped. The exported folder stays separate from model training to preserve
original-source train/validation/test isolation.
