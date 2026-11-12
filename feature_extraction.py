"""Shared training/inference interfaces. Owners must implement before training."""

from functools import lru_cache
from pathlib import Path
import subprocess

import numpy as np


def extract_face_features(path):
    """Maurice: return one 2-D feature row using the training face extractor.

    Use a pretrained face embedding model, reject missing/multiple faces, and
    document model name, version, input normalization and embedding dimensions.
    """
    raise NotImplementedError("Maurice must implement the shared face extractor")


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

    signal = torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0)
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
