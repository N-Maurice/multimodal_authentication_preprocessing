import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from voice_model import (
    build_feature_table,
    calibrate_threshold,
    discover_recordings,
    split_original_recordings,
    train_voice_model,
)


class VoiceModelTests(unittest.TestCase):
    def test_discover_recordings_includes_mp4_audio_containers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_root = Path(temporary_directory)
            speaker_dir = audio_root / "patrick"
            speaker_dir.mkdir()
            recording_path = speaker_dir / "patrick.mp4"
            recording_path.touch()

            recordings = discover_recordings(audio_root)

        self.assertEqual(len(recordings), 1)
        self.assertEqual(recordings[0]["path"], recording_path)

    def test_feature_table_records_augmentation_parameters(self):
        audio = np.sin(np.linspace(0, 20 * np.pi, 16_000)).astype(np.float32)
        recording = {
            "user_id": "kudakwashe",
            "source_id": "kudakwashe/sample",
            "path": Path("sample.ogg"),
            "split": "train",
        }
        with patch("voice_model.load_voice_audio", return_value=(audio, 16_000)):
            features = build_feature_table(
                [recording],
                feature_extractor=lambda waveform, sample_rate: np.array([[1.0, 0.0]]),
            )

        parameters = {
            row["augmentation"]: json.loads(row["augmentation_parameters"])
            for _, row in features.iterrows()
        }
        self.assertEqual(parameters["original"], {})
        self.assertEqual(parameters["pitch_shift_plus_1_semitone"]["n_steps"], 1.0)
        noise = parameters["low_level_noise"]
        self.assertEqual(noise["distribution"], "normal")
        self.assertEqual(noise["clip_range"], [-1.0, 1.0])
        expected_std_dev = max(float(np.sqrt(np.mean(np.square(audio)))) * 0.02, 0.0001)
        self.assertAlmostEqual(noise["std_dev"], expected_std_dev)
        self.assertEqual(
            noise["seed"],
            zlib.crc32(recording["source_id"].encode("utf-8")),
        )

    def test_split_keeps_each_original_in_one_partition(self):
        recordings = [
            {"user_id": user, "source_id": f"{user}/{index}", "path": f"{user}_{index}.ogg"}
            for user in ("kudakwashe", "maurice")
            for index in range(3)
        ]

        split = split_original_recordings(recordings, random_state=7)

        for user in ("kudakwashe", "maurice"):
            user_splits = {row["split"] for row in split if row["user_id"] == user}
            self.assertEqual(user_splits, {"train", "validation", "test"})
        self.assertEqual(len({row["source_id"] for row in split}), len(recordings))

    def test_two_recordings_per_speaker_reserve_validation_without_test_claim(self):
        recordings = [
            {"user_id": user, "source_id": f"{user}/{index}", "path": f"{user}_{index}.ogg"}
            for user in ("kudakwashe", "maurice")
            for index in range(2)
        ]

        split = split_original_recordings(recordings, random_state=7)

        self.assertEqual(sum(row["split"] == "train" for row in split), 2)
        self.assertEqual(sum(row["split"] == "validation" for row in split), 2)
        self.assertFalse(any(row["split"] == "test" for row in split))

    def test_threshold_uses_genuine_and_mismatched_speaker_scores(self):
        labels = np.array(["kudakwashe", "maurice"])
        classes = np.array(["kudakwashe", "maurice"])
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])

        threshold, metrics = calibrate_threshold(labels, probabilities, classes)

        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)
        self.assertEqual(metrics["validation_genuine_trials"], 2)
        self.assertEqual(metrics["validation_impostor_trials"], 2)

    def test_threshold_rejects_mismatched_probability_shape(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(
                ["kudakwashe"],
                np.array([[1.0, 0.0]]),
                np.array(["kudakwashe"]),
            )

    def test_training_saves_predict_proba_model_and_voice_metadata(self):
        feature_table = pd.DataFrame([
            {"user_id": "kudakwashe", "source_id": "kudakwashe/1",
             "split": "train", "augmentation": "original",
             "speaker_embedding_000": 1.0, "speaker_embedding_001": 0.0},
            {"user_id": "kudakwashe", "source_id": "kudakwashe/1",
             "split": "train", "augmentation": "noise",
             "speaker_embedding_000": 0.9, "speaker_embedding_001": 0.1},
            {"user_id": "maurice", "source_id": "maurice/1",
             "split": "train", "augmentation": "original",
             "speaker_embedding_000": 0.0, "speaker_embedding_001": 1.0},
            {"user_id": "maurice", "source_id": "maurice/1",
             "split": "train", "augmentation": "noise",
             "speaker_embedding_000": 0.1, "speaker_embedding_001": 0.9},
            {"user_id": "kudakwashe", "source_id": "kudakwashe/2",
             "split": "validation", "augmentation": "original",
             "speaker_embedding_000": 0.8, "speaker_embedding_001": 0.2},
            {"user_id": "maurice", "source_id": "maurice/2",
             "split": "validation", "augmentation": "original",
             "speaker_embedding_000": 0.2, "speaker_embedding_001": 0.8},
            {"user_id": "kudakwashe", "source_id": "kudakwashe/3",
             "split": "test", "augmentation": "original",
             "speaker_embedding_000": 0.7, "speaker_embedding_001": 0.3},
        ])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({"status": "not_trained"}), encoding="utf-8")

            result = train_voice_model(
                feature_table,
                root / "voice_model.joblib",
                root / "audio_features.csv",
                metadata_path,
            )

            import joblib

            model = joblib.load(root / "voice_model.joblib")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(hasattr(model, "classes_"))
            self.assertEqual(model.predict_proba(np.array([[1.0, 0.0]])).shape, (1, 2))
            self.assertEqual(metadata["status"], "not_trained")
            self.assertEqual(metadata["voice"]["feature_columns"], result["feature_columns"])
            self.assertIsNotNone(metadata["voice"]["threshold"])
            self.assertIsNone(result["test_metrics"])
            self.assertEqual(
                metadata["voice"]["threshold_status"],
                "provisional_validation_only_no_complete_test",
            )
            self.assertEqual(
                metadata["voice"]["test_coverage"]["speakers_missing_test_recordings"],
                ["maurice"],
            )
            self.assertTrue((root / "audio_features.csv").exists())


if __name__ == "__main__":
    unittest.main()
