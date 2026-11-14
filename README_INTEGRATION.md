# Integration notes (Larissa's part)

## Files (drop into the repo root, same paths)
| File | What it is |
|---|---|
| `main.py` | Replaces the scaffold. Face -> request -> voice -> product, with logging and exit codes. |
| `run_demo.py`, `data/processed/demo_scenarios.csv` | Scripted scenarios -> `reports/integration_results.csv`. |
| `data/processed/user_registry.csv` | Filled in: larissa=120, maurice=121, kudakwashe=105, patrick=100 (held-out customers). |
| `data/processed/customer_inference.csv` | The table notebook 01 builds but was never committed. Rebuilt with Patrick's exact logic. |
| `models/product_model.joblib` | Same model re-saved with the current scikit-learn (see blocker 2). |
| `rebuild_product_model.py` | Reproduces the product model and inference table (metrics identical to notebook 01). |
| `models/model_metadata.json` | Only change: top-level `"status": "trained"`. |
| `notebooks/04_system_integration.ipynb` | Integration notebook (unexecuted; run it after the blockers below). |
| `tests/test_flow.py`, `tests/test_cli_smoke.py` | 28 tests with `tests/test_system.py` (all pass). |
| `reports/integration_section.md` | Report text and video script. |

## Blockers before the full demo can run (in this order)
1. **Maurice: commit the face artifacts.** `.gitignore` excludes `models/face_model.joblib` and
   `data/processed/image_features.csv`, so they are not on GitHub, but the assignment requires `image_features.csv` and
   `main.py` needs the face model. Run notebook 02, then `git add -f models/face_model.joblib data/processed/image_features.csv`
   (or delete those two lines from `.gitignore`). The 14 unauthorized portraits are also committed, so confirm consent.
2. **Everyone: same scikit-learn version.** Patrick's original `product_model.joblib` was saved with scikit-learn 1.6.1 and
   fails to load on 1.7 and later (`AttributeError: _RemainderColsList`). The face and voice models were saved with 1.9.
   The re-saved product model in this package loads with 1.9. If anyone still gets a load error, run
   `python rebuild_product_model.py` on their machine.
3. **Record one unauthorized voice** (a non-member saying "Confirm transaction") as
   `data/raw/audio/unauthorized/unauthorized_voice.wav`, otherwise that scenario stays SKIPPED.
4. **Do not re-run notebook 03 yet.** `data/raw/audio` also holds Kudakwashe's `Yes_Approve` / `Confirm_Transaction` clips
   and Patrick's `.mp3` copies, which are not in `audio_features.csv`. Re-running the notebook would pick them up and
   change the splits and every voice number in the report.

## Run
```
pip install -r requirements.txt
python -m unittest discover -s tests   # 28 integration tests pass without the real face model
python main.py                         # interactive demo
python run_demo.py                     # scripted scenarios -> reports/integration_results.csv
```

## What was verified here, and what was not
Verified: all 28 tests; the CLI with the repo's real `FaceAuthenticator`; the real voice model, product model, registry
and scenarios file through `main.attempt` (voice embeddings taken from `audio_features.csv`); the product rebuild
(accuracy 0.3182 / weighted F1 0.2934 / log loss 1.7182, identical to notebook 01).
Not verified: the real face model and the real photo/audio extractors (no face model in the repo, and the sandbox has no
torch/speechbrain/librosa or face weights). Run `python run_demo.py` once on your machine.

## Contribution sheet row (Larissa)
Built the integration CLI (`main.py`: face -> request -> voice -> product, fail-closed denials, access log), the demo
runner and scenarios, the user registry, `rebuild_product_model.py` and `customer_inference.csv`, 13 new tests plus the CLI
smoke tests, `04_system_integration.ipynb`, the integration report section and the video script. Evidence: <commit links>.
