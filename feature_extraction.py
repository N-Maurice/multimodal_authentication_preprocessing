"""Shared training/inference interfaces. Owners must implement before training."""


def extract_face_features(path, weights_dir=None):
    """Return a normalized 1x128 SFace embedding, identical to notebook training.

    Set FACE_WEIGHTS_DIR when deploying the CLI outside the repository. Missing,
    multiple or undersized faces raise FaceInputError (a ValueError).
    """
    import os
    from pathlib import Path
    from face_auth.embeddings import FaceEncoder

    default = Path(__file__).resolve().parent / "models" / "pretrained"
    directory = weights_dir or os.environ.get("FACE_WEIGHTS_DIR", str(default))
    return FaceEncoder(directory).extract(path)


def extract_voice_features(path):
    """Kudakwashe: return one 2-D row using the training audio extractor.

    Document resampling, mono conversion, silence handling and feature order.
    Use pretrained speaker embeddings for verification, with MFCCs for analysis.
    """
    raise NotImplementedError("Kudakwashe must implement the shared voice extractor")
