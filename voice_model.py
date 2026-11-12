"""Training helpers for the speaker-identity classifier."""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from feature_extraction import (
    VOICE_ENCODER_SOURCE,
    VOICE_SAMPLE_RATE,
    extract_voice_features_from_audio,
    load_voice_audio,
)

AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
EMBEDDING_PREFIX = "speaker_embedding_"


def discover_recordings(audio_root):
    """Find labeled recordings in one directory per enrolled speaker."""
    audio_root = Path(audio_root)
    recordings = []
    for user_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        if user_dir.name.lower() == "unauthorized":
            continue
        user_id = user_dir.name.lower()
        for path in sorted(user_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                recordings.append({
                    "user_id": user_id,
                    "source_id": f"{user_id}/{path.stem}",
                    "path": path,
                })
    if not recordings:
        raise ValueError(f"No labeled audio recordings found in {audio_root}")
    return recordings


def split_original_recordings(recordings, random_state=42):
    """Assign each original to exactly one split before any augmentation."""
    by_user = defaultdict(list)
    for recording in recordings:
        by_user[recording["user_id"]].append(dict(recording))

    rng = np.random.default_rng(random_state)
    split_records = []
    for user_id, user_recordings in sorted(by_user.items()):
        if len(user_recordings) < 2:
            raise ValueError(
                f"{user_id} has {len(user_recordings)} original recording(s); "
                "at least two are required for a train/validation split"
            )
        order = rng.permutation(len(user_recordings))
        ordered = [user_recordings[int(index)] for index in order]
        count = len(ordered)
        if count == 2:
            train_count, validation_count = 1, 1
        else:
            validation_count = max(1, int(np.floor(count * 0.2)))
            test_count = max(1, int(np.floor(count * 0.2)))
            train_count = count - validation_count - test_count

        for index, recording in enumerate(ordered):
            if index < train_count:
                recording["split"] = "train"
            elif index < train_count + validation_count:
                recording["split"] = "validation"
            else:
                recording["split"] = "test"
            split_records.append(recording)

    return split_records


def augment_audio(audio, sample_rate, seed):
    """Create pitch-shift and low-level noise variants without changing labels."""
    import librosa

    rng = np.random.default_rng(seed)
    pitch_shifted = librosa.effects.pitch_shift(
        audio,
        sr=sample_rate,
        n_steps=1.0,
    )
    rms = float(np.sqrt(np.mean(np.square(audio))))
    noise_scale = max(rms * 0.02, 0.0001)
    noisy = audio + rng.normal(0.0, noise_scale, size=audio.shape)
    return [
        ("pitch_shift_plus_1_semitone", pitch_shifted.astype(np.float32)),
        ("low_level_noise", np.clip(noisy, -1.0, 1.0).astype(np.float32)),
    ]


def analyze_audio(audio, sample_rate=VOICE_SAMPLE_RATE):
    """Summarize MFCC, spectral roll-off and RMS for analysis and CSV export."""
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    rms = librosa.feature.rms(y=audio)[0]
    features = {}
    for index, coefficients in enumerate(mfcc, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficients))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficients))
    features.update({
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    })
    return features


def build_feature_table(recordings, feature_extractor=extract_voice_features_from_audio):
    """Extract embeddings and analysis features, keeping variants with sources."""
    rows = []
    for recording in recordings:
        audio, sample_rate = load_voice_audio(recording["path"])
        variants = [("original", audio)] + augment_audio(
            audio,
            sample_rate,
            seed=zlib.crc32(recording["source_id"].encode("utf-8")),
        )
        for augmentation, waveform in variants:
            embedding = feature_extractor(waveform, sample_rate).reshape(-1)
            row = {
                "user_id": recording["user_id"],
                "source_id": recording["source_id"],
                "split": recording["split"],
                "augmentation": augmentation,
            }
            row.update(analyze_audio(waveform, sample_rate))
            row.update({
                f"{EMBEDDING_PREFIX}{index:03d}": float(value)
                for index, value in enumerate(embedding)
            })
            rows.append(row)
    return pd.DataFrame(rows)


def calibrate_threshold(labels, probabilities, classes):
    """Choose a validation threshold maximizing balanced genuine/impostor accuracy."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), len(classes)):
        raise ValueError("Probability shape must match labels and classifier classes")
    if len(labels) == 0:
        raise ValueError("At least one validation recording is required")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have a positive sum")
    probabilities = probabilities / row_sums

    class_indices = {label: index for index, label in enumerate(classes)}
    genuine_scores = []
    impostor_scores = []
    for row_index, label in enumerate(labels):
        if label not in class_indices:
            raise ValueError(f"Validation label {label!r} is not a classifier class")
        genuine_index = class_indices[label]
        genuine_scores.append(probabilities[row_index, genuine_index])
        impostor_scores.extend(
            probabilities[row_index, index]
            for index in range(len(classes))
            if index != genuine_index
        )
    if not impostor_scores:
        raise ValueError("At least two enrolled speakers are needed to calibrate rejection")

    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    candidates = np.unique(np.concatenate(([0.0, 1.0], genuine_scores, impostor_scores)))
    best = None
    for threshold in candidates:
        false_accept_rate = float(np.mean(impostor_scores >= threshold))
        false_reject_rate = float(np.mean(genuine_scores < threshold))
        balanced_accuracy = 1.0 - (false_accept_rate + false_reject_rate) / 2.0
        candidate = (balanced_accuracy, -false_accept_rate, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2], {
        "validation_false_accept_rate": -best[1],
        "validation_false_reject_rate": float(np.mean(genuine_scores < best[2])),
        "validation_genuine_trials": int(len(genuine_scores)),
        "validation_impostor_trials": int(len(impostor_scores)),
    }


def evaluate_predictions(labels, probabilities, classes, threshold):
    """Return identity and known-speaker mismatch metrics for one split."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), len(classes)):
        raise ValueError("Probability shape must match labels and classifier classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have a positive sum")
    probabilities = probabilities / row_sums
    predicted = classes[np.argmax(probabilities, axis=1)]
    genuine_scores = []
    impostor_scores = []
    class_indices = {label: index for index, label in enumerate(classes)}
    for row_index, label in enumerate(labels):
        if label not in class_indices:
            raise ValueError(f"Evaluation label {label!r} is not a classifier class")
        genuine_index = class_indices[label]
        genuine_scores.append(probabilities[row_index, genuine_index])
        impostor_scores.extend(
            probabilities[row_index, index]
            for index in range(len(classes))
            if index != genuine_index
        )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "false_accept_rate": float(np.mean(np.asarray(impostor_scores) >= threshold)),
        "false_reject_rate": float(np.mean(np.asarray(genuine_scores) < threshold)),
        "recordings": int(len(labels)),
    }


def train_voice_model(feature_table, model_path, features_path, metadata_path):
    """Fit the classifier on training sources and calibrate only on validation sources."""
    import joblib

    embedding_columns = sorted(
        column for column in feature_table.columns
        if column.startswith(EMBEDDING_PREFIX)
    )
    if not embedding_columns:
        raise ValueError("Feature table does not contain speaker embeddings")

    training = feature_table[
        (feature_table["split"] == "train")
    ]
    validation = feature_table[
        (feature_table["split"] == "validation")
        & (feature_table["augmentation"] == "original")
    ]
    test = feature_table[
        (feature_table["split"] == "test")
        & (feature_table["augmentation"] == "original")
    ]
    if training["user_id"].nunique() < 2:
        raise ValueError("At least two speakers are needed to train the classifier")
    if validation.empty:
        raise ValueError("No validation recordings are available for threshold calibration")
    if validation["user_id"].nunique() != training["user_id"].nunique():
        raise ValueError("Every training speaker must have a validation recording")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(
        training[embedding_columns].to_numpy(dtype=np.float32),
        training["user_id"],
    )

    validation_probabilities = model.predict_proba(
        validation[embedding_columns].to_numpy(dtype=np.float32)
    )
    threshold, calibration = calibrate_threshold(
        validation["user_id"].to_numpy(),
        validation_probabilities,
        model.classes_,
    )
    validation_metrics = evaluate_predictions(
        validation["user_id"].to_numpy(),
        validation_probabilities,
        model.classes_,
        threshold,
    )
    test_metrics = None
    if not test.empty:
        test_probabilities = model.predict_proba(
            test[embedding_columns].to_numpy(dtype=np.float32)
        )
        test_metrics = evaluate_predictions(
            test["user_id"].to_numpy(),
            test_probabilities,
            model.classes_,
            threshold,
        )

    features_path = Path(features_path)
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(features_path, index=False)
    joblib.dump(model, model_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    voice_metadata = metadata.setdefault("voice", {})
    voice_metadata.update({
        "owner": "Kudakwashe",
        "threshold": threshold,
        "threshold_status": (
            "validation_calibrated_with_independent_test"
            if test_metrics is not None
            else "provisional_validation_only_no_independent_test"
        ),
        "feature_columns": embedding_columns,
        "pretrained_model": VOICE_ENCODER_SOURCE,
        "pretrained_model_license": "Apache-2.0",
        "classifier": "scikit-learn LogisticRegression",
        "sample_rate_hz": VOICE_SAMPLE_RATE,
        "embedding_dimension": len(embedding_columns),
        "embedding_normalization": "L2",
        "silence_trim_top_db": 30,
        "source_counts_by_split": {
            split: int(rows["source_id"].nunique())
            for split, rows in feature_table.groupby("split")
        },
        "validation_calibration": calibration,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "model": model,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "calibration": calibration,
        "feature_columns": embedding_columns,
    }
