# System integration and demonstration (Larissa)

Paste into `final_report.md` / the PDF as the "System demonstration" section. Fill the `<...>` fields from a real run of
`python run_demo.py` (saved in `reports/integration_results.csv`); do not type numbers the code did not produce.

## Design
`main.py` runs the three checkpoints from the assignment diagram in order:

1. **Face check.** `extract_face_features()` (YuNet + SFace, the same function used in training) turns the photo into a
   128-number embedding. The face model (`FaceAuthenticator`) accepts it only if the classifier probability is at least
   0.5 **and** the cosine similarity to the predicted member's average embedding reaches `face.threshold` (0.417). A
   rejected face scores zero, and the CLI denies it. An unreadable image, a photo with no face or with several faces is
   also a denial. The voice file is not processed after a face failure.
2. **Prediction request.** The recognised user asks for a product prediction.
3. **Voice check.** `extract_voice_features()` (ECAPA-TDNN speaker embedding) is scored by the voice model. The
   probability must reach `voice.threshold` (0.285) **and** name the same registered user as the face. A valid voice of
   a different member is denied at the identity-match step. Silent or undecodable audio is denied.

Only after both checks does the product model (Random Forest pipeline) receive the verified customer's single
pre-visit feature row and return a recommendation. Every attempt is appended to `reports/access_log.csv`. Exit codes:
0 granted, 1 denied, 2 setup error. The registry (`user_registry.csv`) maps each consenting member to a customer the
product model did not see in training.

## Model results (from the team notebooks and `model_metadata.json`)
| Model | Data | Accuracy | F1 | Log loss | Notes |
|---|---|---|---|---|---|
| Face (SFace + logistic regression + cosine gate) | 4 test photos | 100 % | 1.00 (macro) | 0.068 | 14 of 14 strangers denied; leave-one-member-out: 0 of 12 impostors accepted |
| Voice (ECAPA-TDNN + logistic regression) | 4 validation recordings | 75 % | 0.67 (macro) | 1.13 | validation only, no test set; 1 of 12 mismatch trials accepted; no unknown-speaker recordings |
| Product (Random Forest, grouped by customer) | 22 held-out transactions | 31.8 % | 0.29 (weighted) | 1.72 | random chance is 20 % |

## Scenario results
| Scenario | Face | Voice | Expected | Actual |
|---|---|---|---|---|
| authorized (4 members) | test photo | same member | granted, product shown | <from integration_results.csv> |
| unknown face | stranger photo | member | denied at face check | <...> |
| blank image | white image | member | denied at face check | <...> |
| unknown voice | member | stranger recording | denied at voice check | <...> |
| silent audio | member | silence | denied at voice check | <...> |
| mismatched voice | member A | member B | denied at identity-match check | <...> |

Summary: <n> attempts, <n> correct grants, <n> false accepts. Automated tests: <n> pass (`python -m unittest discover -s tests`).

## Known weaknesses (checked on the stored embeddings, notebook 04 section 5)
- The voice model's probabilities are low (for four speakers 0.25 is chance) and its threshold is only just above chance.
  Of 8 original recordings, 7 are identified correctly (3 of 4 on validation: Maurice's second recording is identified as
  Patrick). Presenting Patrick's face with that Maurice recording would be accepted (1 of 12 mismatch trials).
- The authorized demo recordings were also used to train or calibrate the voice model, because there are only two per
  speaker. They show that the flow works, not how accurate it is on new recordings.
- The recommendation is only modestly better than chance (31.8 % vs 20 %).
- No liveness detection: a printed photo or replayed recording could be presented. The customer links are a simulation
  choice.

---

# Video script (about 3-4 minutes)
Record the terminal (and the notebook for the last step). Use the test photos, not the training photos.

| Time | Show | Say |
|---|---|---|
| 0:00 | Repo tree | Team, goal: face then voice verification before a product recommendation. |
| 0:20 | Flow diagram | The three checkpoints; unknowns are rejected at each one. |
| 0:40 | `python main.py` (prompts) | Enter an authorized face path. It shows "Face recognised", then asks to request a prediction. |
| 1:10 | Enter the same member's voice path | "Voice approved", then "ACCESS GRANTED, Recommended product: ...". |
| 1:40 | Run with an **unauthorized photo** + a valid voice | Stops at the face check; the voice file is never processed. |
| 2:10 | Run with a valid face + the **unauthorized voice** | Denied at the voice check, no product. |
| 2:30 | Face of member A + voice of member B | Denied at the identity-match check. |
| 2:50 | `type reports\access_log.csv` (or `cat`) | Every attempt is logged; denied ones carry no prediction. |
| 3:10 | `python run_demo.py` and the notebook results table | All scenarios behave as expected. |
| 3:30 | Contributions + limitations | Who built what; weak voice model; no liveness; small data. |

Commands (quote paths that contain spaces):
```
python main.py --face data/raw/images/larissa/Larissa_Surprised.jpeg --voice "data/raw/audio/larissa/Larissa_Confirm transaction.ogg"
python main.py --face data/raw/images/unauthorized/unauthorized_01.jpeg --voice "data/raw/audio/larissa/Larissa_Confirm transaction.ogg"
python main.py --face data/raw/images/larissa/Larissa_Surprised.jpeg --voice "data/raw/audio/maurice/Maurice_Approved.ogg"
python run_demo.py
python -m unittest discover -s tests
```
