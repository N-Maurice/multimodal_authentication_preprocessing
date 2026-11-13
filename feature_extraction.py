"""Face feature extraction shared by notebook training and the authentication CLI.

Pretrained face encoder
-----------------------
* Detector:   YuNet, OpenCV Zoo release ``face_detection_yunet_2023mar.onnx``
* Encoder:    SFace, OpenCV Zoo release ``face_recognition_sface_2021dec.onnx``
              (Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for
              Robust Face Recognition", 2021)
* Embedding:  128 float32 values, L2-normalised (length 1)

Both ONNX files are downloaded once into ``models/pretrained/`` by the notebook
(or supplied offline). Their SHA-256 hashes are recorded in
``models/model_metadata.json``.
"""
from functools import lru_cache
import os
from pathlib import Path

PRETRAINED_MODEL = "OpenCV Zoo SFace (face_recognition_sface_2021dec) with YuNet detector (face_detection_yunet_2023mar)"
ENCODER_NAME = "SFace"
ENCODER_VERSION = "2021dec"
DETECTOR_NAME = "YuNet"
DETECTOR_VERSION = "2023mar"
EMBEDDING_SIZE = 128
FEATURE_COLUMNS = [f"embedding_{index:03d}" for index in range(EMBEDDING_SIZE)]
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "models" / "pretrained"

from functools import lru_cache
from pathlib import Path
import subprocess

import numpy as np


@lru_cache(maxsize=4)
def _encoder(weights_dir):
    from face_auth.embeddings import FaceEncoder
    return FaceEncoder(weights_dir)


def extract_face_features(path, weights_dir=None):
    """Return the face embedding of one photograph as a 2-D array of shape (1, 128).

    Steps: decode the image with EXIF rotation, shrink it to at most 1200 pixels
    on the longest side, detect faces with YuNet, require **exactly one** face
    (confidence >= 0.8, box side >= 30 px), align it to 112x112 with the five
    landmarks and embed it with SFace.

    Raises ``face_auth.embeddings.FaceInputError`` (a ``ValueError``) when the file
    cannot be read, contains no face, contains several faces, or the face is too
    small. Set ``FACE_WEIGHTS_DIR`` to use weights outside the repository.
    """
    directory = weights_dir or os.environ.get("FACE_WEIGHTS_DIR", str(DEFAULT_WEIGHTS_DIR))
    features = _encoder(str(directory)).extract(path)
    if features.shape != (1, EMBEDDING_SIZE):
        raise ValueError(f"Expected one row of {EMBEDDING_SIZE} features, got {features.shape}")
    return features


VOICE_SAMPLE_RATE = 16_000
VOICE_ENCODER_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def load_voice_audio(path, sample_rate=VOICE_SAMPLE_RATE):
    """Decode mono audio at the encoder rate and trim leading/trailing silence."""
    import librosa
    import imageio_ffmpeg

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    process = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Unable to decode audio {path}: {detail}")
    audio = np.frombuffer(process.stdout, dtype="<f4")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio contains no usable samples: {path}")

    audio, _ = librosa.effects.trim(audio, top_db=30)
    if audio.size == 0 or float(np.max(np.abs(audio))) <= 1e-8:
        raise ValueError(f"Audio contains no speech after silence trimming: {path}")
    return np.asarray(audio, dtype=np.float32), sample_rate


@lru_cache(maxsize=1)
def _load_voice_encoder():
    from speechbrain.inference.speaker import EncoderClassifier

    cache_dir = Path.home() / ".cache" / "multimodal-authentication" / "ecapa-voxceleb"
    return EncoderClassifier.from_hparams(
        source=VOICE_ENCODER_SOURCE,
        savedir=str(cache_dir),
        run_opts={"device": "cpu"},
    )


def extract_voice_features_from_audio(audio, sample_rate=VOICE_SAMPLE_RATE):
    """Return one L2-normalized ECAPA-TDNN speaker embedding as a 2-D row."""
    import librosa
    import torch

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain finite, non-empty samples")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if sample_rate != VOICE_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=VOICE_SAMPLE_RATE,
        )
    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    if waveform.size == 0 or float(np.max(np.abs(waveform))) <= 1e-8:
        raise ValueError("Audio contains no speech after silence trimming")

    signal = torch.from_numpy(
        np.array(waveform, dtype=np.float32, order="C", copy=True)
    ).unsqueeze(0)
    with torch.inference_mode():
        embedding = _load_voice_encoder().encode_batch(signal)
    embedding = embedding.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("Speaker encoder returned non-finite features")
    norm = float(np.linalg.norm(embedding))
    if norm == 0:
        raise ValueError("Speaker encoder returned a zero-length embedding")
    return embedding / norm


def extract_voice_features(path):
    """Extract one 2-D 192-D speaker row using the shared inference pipeline.

    Audio is decoded, converted to mono, resampled to 16 kHz, and trimmed at
    30 dB below peak. SpeechBrain's pretrained ECAPA-TDNN model supplies the
    L2-normalized embedding used by both training and authentication.
    """
    audio, sample_rate = load_voice_audio(path)
    return extract_voice_features_from_audio(audio, sample_rate)
