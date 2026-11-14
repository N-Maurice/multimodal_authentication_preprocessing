"""Sequential-flow tests for main.attempt and run_demo using fake models."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import main
import run_demo


class Model:
    def __init__(self, classes, probabilities):
        self.classes_, self.probabilities, self.calls = classes, probabilities, 0

    def predict_proba(self, features):
        self.calls += 1
        return np.array([self.probabilities])


class Product:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return ["Books"]

    def predict_proba(self, frame):
        return np.array([[.6, .4]])


def make_ctx(face, voice, extract_face=None, extract_voice=None):
    calls = {"face": 0, "voice": 0}

    def default_face(_):
        calls["face"] += 1
        return [[0]]

    def default_voice(_):
        calls["voice"] += 1
        return [[0]]

    return {
        "metadata": {"face": {"threshold": .4}, "voice": {"threshold": .8},
                     "product": {"feature_columns": ["x"]}},
        "face_gate": .5,
        "registry": {"larissa": "101", "maurice": "114"},
        "names": {"larissa": "Larissa", "maurice": "Maurice"},
        "customers": pd.DataFrame({"customer_id": ["101", "114"], "x": [1, 2]}),
        "face_model": face, "voice_model": voice, "product_model": Product(),
        "extract_face": extract_face or default_face,
        "extract_voice": extract_voice or default_voice,
    }, calls


class FlowTests(unittest.TestCase):
    def test_full_transaction_is_granted_and_predicts_once(self):
        ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]))
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"], r["identity"], r["prediction"]),
                         ("granted", "done", "larissa", "Books"))
        self.assertEqual(r["product_confidence"], .6)
        self.assertEqual(ctx["product_model"].calls, 1)

    def test_face_model_zero_scores_mean_rejected(self):
        # FaceAuthenticator returns all zeros for a rejected face.
        ctx, calls = make_ctx(Model(["larissa", "maurice"], [0., 0.]), Model(["larissa"], [1.]))
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"]), ("denied", "face"))
        self.assertEqual(calls["voice"], 0)
        self.assertEqual(ctx["product_model"].calls, 0)

    def test_unknown_face_never_reaches_voice_or_product(self):
        ctx, calls = make_ctx(Model(["larissa", "maurice"], [.55, .45]), Model(["larissa"], [1.]))
        ctx["face_gate"] = .8
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"], r["prediction"]), ("denied", "face", ""))
        self.assertEqual(calls["voice"], 0)  # voice file is not even processed
        self.assertEqual(ctx["voice_model"].calls, 0)
        self.assertEqual(ctx["product_model"].calls, 0)

    def test_unusable_face_image_is_denied_not_a_crash(self):
        def no_face(_):
            raise ValueError("no face found")
        ctx, calls = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]), extract_face=no_face)
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"]), ("denied", "face"))
        self.assertEqual(calls["voice"], 0)

    def test_unusable_voice_sample_is_denied_not_a_crash(self):
        def silent(_):
            raise ValueError("no speech")
        ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]), extract_voice=silent)
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"], r["prediction"]), ("denied", "voice", ""))
        self.assertEqual(ctx["product_model"].calls, 0)

    def test_unknown_voice_is_denied_without_prediction(self):
        ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["larissa", "maurice"], [.5, .5]))
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"]), ("denied", "voice"))
        self.assertEqual(ctx["product_model"].calls, 0)

    def test_other_enrolled_voice_is_denied_as_mismatch(self):
        ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["maurice"], [.99]))
        r = main.attempt(ctx, "f", "v")
        self.assertEqual((r["outcome"], r["stage"]), ("denied", "match"))
        self.assertEqual(ctx["product_model"].calls, 0)

    def test_request_step_happens_between_face_and_voice(self):
        ctx, calls = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]))
        seen = []
        main.attempt(ctx, "f", "v", pause=lambda name: seen.append((name, dict(calls))))
        self.assertEqual(seen, [("Larissa", {"face": 1, "voice": 0})])

    def test_missing_inference_row_is_a_setup_error(self):
        ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]))
        ctx["customers"] = ctx["customers"].iloc[1:]
        with self.assertRaises(ValueError):
            main.attempt(ctx, "f", "v")

    def test_denied_attempt_is_logged_without_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "reports" / "access_log.csv"
            main.log_attempt(log, "a.jpeg", "b.ogg",
                             {"outcome": "denied", "stage": "face", "identity": "", "reason": "x"})
            with log.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((rows[0]["outcome"], rows[0]["stopped_at"]), ("denied", "face"))


class DemoRunnerTests(unittest.TestCase):
    def test_scenarios_pass_fail_and_skip_are_reported_honestly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "f.txt").write_text("x")
            (root / "v.txt").write_text("x")
            ctx, _ = make_ctx(Model(["larissa"], [.95]), Model(["larissa"], [.9]))
            scenarios = pd.DataFrame([
                ("ok", "f.txt", "v.txt", "granted", "done"),
                ("wrong_expectation", "f.txt", "v.txt", "denied", "face"),
                ("missing", "nope.txt", "v.txt", "denied", "face"),
            ], columns=["scenario", "face", "voice", "expect_outcome", "expect_stage"])
            table = run_demo.run_scenarios(ctx, scenarios, root=root)
            self.assertEqual(table.status.tolist(), ["PASS", "FAIL", "SKIPPED"])

    def test_special_files_are_created(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(run_demo.make_special("@silence", folder).exists())
            self.assertTrue(run_demo.make_special("@blank_image", folder).exists())
            with self.assertRaises(ValueError):
                run_demo.make_special("@nope", folder)


if __name__ == "__main__":
    unittest.main()
