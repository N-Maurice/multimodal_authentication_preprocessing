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


def extract_voice_features(path):
    """Kudakwashe: return one 2-D row using the training audio extractor.

    Document resampling, mono conversion, silence handling and feature order.
    Use pretrained speaker embeddings for verification, with MFCCs for analysis.
    """
    raise NotImplementedError("Kudakwashe must implement the shared voice extractor")
