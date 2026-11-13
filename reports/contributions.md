# Team contributions

| Member | Ownership | Outputs | Evidence / commits |
|---|---|---|---|
| Patrick | 01_product_recommendation.ipynb | merged_dataset.csv, customer_inference.csv, product_model.joblib | Pending |
| Maurice | 02_face_recognition.ipynb, face extractor | image_features.csv, face_model.joblib | Pending |
| Kudakwashe | 03_voice_verification.ipynb, voice extractor | audio_features.csv, voice_model.joblib | Executed notebook and passing tests; work is on `feature/voice` |
| Larissa | 04_system_integration.ipynb, main.py | CLI, registry, integration evaluation, video | Pending |

## Kudakwashe voice contribution

Completed the voice feature extraction, augmentation, speaker-classifier training,
and validation workflow on 2026-11-13. The executed notebook produced
[audio_features.csv](../data/processed/audio_features.csv) and
[voice_model.joblib](../models/voice_model.joblib). The feature table has 21 rows
from seven original recordings across three speakers; each source has two
augmentation variants.

The provisional validation results were accuracy 0.33, macro F1 0.22, log loss
1.19, false-accept rate 0.17, and false-reject rate 0.67. These are not reliable
performance estimates: only Kudakwashe has a test recording, so a complete
independent test result is unavailable. No unknown-speaker recordings were
available to measure unknown-person false acceptance. The full test suite passed
(12 tests). The work is tracked on `feature/voice`; add its commit link if required
for submission.

Record other members' actual work, dates, commit links and review contributions
before submission.
