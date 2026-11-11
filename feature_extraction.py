"""Face feature extraction shared by training and authentication inference."""


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
