"""End-to-end run of the real CLI with tiny saved models and fake feature extractors.

Uses the repository's real FaceAuthenticator (all-zero scores for rejected faces), a real
scikit-learn voice classifier and a real preprocessing+classifier product pipeline.
Checks exit codes (0 granted, 1 denied, 2 setup error) and that denied runs print no product.
Skipped automatically when the dependencies are not installed.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from face_auth.pipeline import FaceAuthenticator
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

ROOT = Path(__file__).resolve().parents[1]
FAKE_EXTRACTORS = '''\
import numpy as np

def _unit(dim, index):
    v = np.zeros(dim); v[index] = 1.0
    return v

def extract_face_features(path):
    name = open(path).read().strip()
    if name == "noface":
        raise ValueError("no face found")
    index = {"larissa": 0, "maurice": 1, "stranger": 5}[name]
    return (_unit(128, index) + 0.01 * np.arange(128) / 128).reshape(1, -1)

def extract_voice_features(path):
    name = open(path).read().strip()
    if name == "silent":
        raise ValueError("no speech")
    vectors = {"larissa": _unit(8, 0), "maurice": _unit(8, 1),
               "stranger": (_unit(8, 0) + _unit(8, 1)) / 2 ** .5}
    return vectors[name].reshape(1, -1)
'''


@unittest.skipUnless(HAVE_DEPS, "scikit-learn/joblib/face_auth not importable")
class CliSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        t = cls.tmp
        (t / "models").mkdir()
        (t / "data/processed").mkdir(parents=True)
        shutil.copy(ROOT / "main.py", t / "main.py")
        shutil.copytree(ROOT / "face_auth", t / "face_auth")
        (t / "feature_extraction.py").write_text(FAKE_EXTRACTORS)
        rng = np.random.default_rng(0)

        def cluster(dim, index, n=6):
            base = np.zeros(dim); base[index] = 1.0
            return base + 0.05 * rng.standard_normal((n, dim))

        Xf = np.vstack([cluster(128, 0), cluster(128, 1)]); y = ["larissa"] * 6 + ["maurice"] * 6
        face = FaceAuthenticator(C=100, similarity_threshold=.5, confidence_threshold=.5).fit(Xf, y)
        joblib.dump(face, t / "models/face_model.joblib")
        Xv = np.vstack([cluster(8, 0), cluster(8, 1)])
        joblib.dump(LogisticRegression(C=100).fit(Xv, y), t / "models/voice_model.joblib")
        frame = pd.DataFrame({"f1": [1., 2., 3., 4.], "f2": ["a", "b", "a", "b"]})
        product = Pipeline([("pre", ColumnTransformer([("n", StandardScaler(), ["f1"]),
                                                       ("c", OneHotEncoder(), ["f2"])])),
                            ("clf", LogisticRegression())]).fit(frame, ["Books", "Sports", "Books", "Sports"])
        joblib.dump(product, t / "models/product_model.joblib")
        (t / "models/model_metadata.json").write_text(json.dumps({
            "status": "trained",
            "face": {"threshold": .5, "confidence_threshold": .5},
            "voice": {"threshold": .9},
            "product": {"feature_columns": ["f1", "f2"]}}))
        (t / "data/processed/user_registry.csv").write_text(
            "user_id,display_name,customer_id\nlarissa,Larissa,120\nmaurice,Maurice,121\n")
        (t / "data/processed/customer_inference.csv").write_text(
            "customer_id,f1,f2\n120,1,a\n121,4,b\n")
        for name in ("larissa", "maurice", "stranger", "noface", "silent"):
            (t / f"{name}.txt").write_text(name)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, face, voice):
        return subprocess.run(
            [sys.executable, "main.py", "--yes", "--face", f"{face}.txt", "--voice", f"{voice}.txt"],
            cwd=self.tmp, capture_output=True, text=True)

    def test_granted(self):
        r = self.cli("larissa", "larissa")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Recommended product: Books", r.stdout)

    def test_unknown_face_denied_by_real_face_model(self):
        r = self.cli("stranger", "larissa")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("DENIED at the face check", r.stdout)
        self.assertNotIn("Recommended product", r.stdout)

    def test_image_without_face_denied(self):
        r = self.cli("noface", "larissa")
        self.assertEqual(r.returncode, 1)
        self.assertIn("DENIED at the face check", r.stdout)

    def test_unknown_voice_denied(self):
        r = self.cli("larissa", "stranger")
        self.assertEqual(r.returncode, 1)
        self.assertIn("DENIED at the voice check", r.stdout)

    def test_silent_audio_denied(self):
        r = self.cli("larissa", "silent")
        self.assertEqual(r.returncode, 1)
        self.assertIn("DENIED at the voice check", r.stdout)
        self.assertNotIn("Recommended product", r.stdout)

    def test_other_member_voice_denied(self):
        r = self.cli("larissa", "maurice")
        self.assertEqual(r.returncode, 1)
        self.assertIn("DENIED at the identity-match check", r.stdout)
        self.assertNotIn("Recommended product", r.stdout)

    def test_untrained_metadata_is_setup_error(self):
        meta = self.tmp / "models/model_metadata.json"
        original = meta.read_text()
        try:
            meta.write_text(json.dumps({"status": "not_trained"}))
            self.assertEqual(self.cli("larissa", "larissa").returncode, 2)
        finally:
            meta.write_text(original)

    def test_missing_face_model_is_setup_error_with_owner_hint(self):
        model = self.tmp / "models/face_model.joblib"
        backup = self.tmp / "face_model.bak"
        shutil.move(model, backup)
        try:
            r = self.cli("larissa", "larissa")
            self.assertEqual(r.returncode, 2)
            self.assertIn("Maurice", r.stderr)
        finally:
            shutil.move(backup, model)

    def test_access_log_written(self):
        self.cli("larissa", "larissa")
        self.assertTrue((self.tmp / "reports/access_log.csv").exists())


if __name__ == "__main__":
    unittest.main()
