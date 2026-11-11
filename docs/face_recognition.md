# Face authentication workflow

Owner: Maurice. Development branch: `feature/face`.

Start with [the complete notebook](../notebooks/02_face_recognition.ipynb).
Run every cell in order; configuration is at the top. The notebook performs real
training, evaluation and export, rather than supplying placeholder metrics.

## Navigation

| Location | Purpose |
|---|---|
| `notebooks/02_face_recognition.ipynb` | Narrative, analysis, filters, training, validation, testing and export |
| `face_auth/embeddings.py` | YuNet detection, alignment, pretrained SFace features and explicit weight downloads |
| `face_auth/pipeline.py` | Data audit, safe splits, training augmentation, classifier and rejection calibration |
| `face_auth/__main__.py` | Face-only inference CLI |
| `feature_extraction.py` | Shared face extractor for full system integration |
| `requirements/face.txt` | Face-only pip environment |
| `environment-face.yml` | Anaconda environment |
| `tests/test_face_pipeline.py` | Data isolation, duplicate detection, rejection and serialization tests |

## Local Anaconda

```bash
conda env create -f environment-face.yml
conda activate multimodal-face
python -m ipykernel install --user --name multimodal-face --display-name "Python (multimodal-face)"
jupyter lab
```

Select the `Python (multimodal-face)` kernel. For an existing environment, use
`python -m pip install -r requirements/face.txt` with that environment activated.
The notebook prints its Python executable so you can verify the active kernel.
No GPU or CUDA dependencies are needed. Do not install multiple OpenCV variants
in the same environment; headless OpenCV works with Matplotlib in notebooks.

## Colab

Upload or clone the full repository into `/content`, including the package and
images. Keep biometric uploads private. Open the notebook, enable
`INSTALL_MISSING=True` if needed, then run the setup cells. Restart the kernel
if package installation requests it. For Drive, mount it with Colab's normal
Drive controls and set `PROJECT_ROOT_OVERRIDE` to the repository directory.
Set `OUTPUT_ROOT_OVERRIDE` to a writable Drive path if outputs must survive the
runtime. Download artifacts before deleting a temporary runtime.

## Kaggle

Attach a private dataset containing the complete repository and another for
images if desired. Set `PROJECT_ROOT_OVERRIDE` to the folder containing
`face_auth/` and `IMAGE_DIR_OVERRIDE` to the folder containing user directories.
Outputs default to `/kaggle/working/multimodal-face`. Enable Internet for weight
downloads; offline runs require both ONNX files in `WEIGHTS_DIR_OVERRIDE` and
`DOWNLOAD_WEIGHTS=False`. Dependencies must already be installed or supplied as
offline wheels when Internet is disabled. The notebook does not write to inputs.

## Data and evaluation policy

Use `data/raw/images/<user_id>/<filename>`. Each enrolled identity needs at least
three independent, detectable photographs. The current three-image identities
receive one training, one validation and one test capture. Underrepresented
identities are held out completely as unknown test users. This means Kudakwashe
is currently an unknown test person, not enrolled; add independent captures to
enroll that identity on a future run. Patrick still needs images.

Original-source hashes prevent exact duplicate leakage. Review near duplicates
and capture sessions manually. Augmentation only follows the split and applies
to training crops. It does not create independent evaluation examples.

Logistic regression regularization is selected on validation. A cosine threshold
is calibrated from genuine versus wrong-identity validation comparisons. These
are proxy comparisons, not measurements on an unknown-person validation cohort.
The test set is untouched until the final evaluation; the saved model is not
refit afterward. Add many more sessions and separate unknown validation/test
cohorts before making claims about real-world biometric performance.

## Generated outputs

- `data/processed/face_manifest.csv`: audit and source partitions.
- `data/processed/image_features.csv`: provenance and 128 numeric embedding columns.
- `models/face_model.joblib`: trained classifier, class centroids and rejection settings.
- `models/pretrained/*.onnx`: inference dependencies; downloaded explicitly.
- `models/model_metadata.json`: face settings, classes, versions, weight hashes and evaluation.
- `reports/face/`: data audit, plots, calibration curve, test attempts and metrics.

Generated biometric files and weights are ignored by Git. Notebook outputs also
contain participant images: clear outputs before sharing publicly if consent
has not been granted. Model hashes and dependency versions support reproduction;
use the same saved ONNX files because upstream URLs can change.

## Inference and integration

After notebook export, from the repository root:

```bash
python -m face_auth --image /path/to/new-photo.jpg
# For cloud or relocated outputs:
python -m face_auth --image /path/to/new-photo.jpg \
  --model /path/to/output/models/face_model.joblib \
  --weights /path/to/onnx-directory
python -m pytest -q
```

Exit codes: 0 verified face, 1 denied input/identity, 2 setup failure. Missing,
multiple or too-small faces are rejected. Unknown embeddings must pass both the
predicted identity's cosine gate and classifier confidence gate.

`FaceAuthenticator.predict_proba` returns zero scores for rejected faces; those
rows intentionally do not sum to one. The full CLI rejects them through its
positive confidence threshold. Use `classifier_` probabilities for log loss.
The notebook preserves product/voice metadata and global `status`; it does not
mark the whole application trained. Voice implementation and registry mapping
are still required. Set `FACE_WEIGHTS_DIR` when relocating the shared extractor.
The face-only CLI does not approve a transaction.

Only load trusted joblib files. This classroom workflow has no liveness or
replay defense. SFace/YuNet sources, methodology and upstream license links are
included in the notebook.

## Verified initial run

The workflow was executed end to end on the seven supplied originals. Two
identities (Larissa and Maurice) were enrolled; Kudakwashe's single photograph
was withheld entirely as an unknown test identity. The split contained two
training originals (18 augmented feature rows), two validation originals, two
known test originals and one unknown test original.

On that small test set: closed-set accuracy 1.0, macro F1 1.0, log loss 0.1978;
both known identities were accepted correctly and the one unknown attempt was
rejected. These counts are too small for a reliable performance estimate.

The executed notebook is saved locally at
`reports/face/02_face_recognition.executed.ipynb`; the version-controlled notebook
has clean outputs to avoid embedding participant photographs in its history.
Seventeen automated tests passed, face-only CLI success/denial exit codes were
checked, and a second offline notebook run passed with separate project,
working, weight and output directories. Actual hosted Colab/Kaggle sessions have
not been run here; their configuration and writable-path behavior are supported
and were checked locally.
