"""Shared training/inference interfaces. Owners must implement before training."""


def extract_face_features(path):
    """Maurice: return one 2-D feature row using the training face extractor.

    Use a pretrained face embedding model, reject missing/multiple faces, and
    document model name, version, input normalization and embedding dimensions.
    """
    raise NotImplementedError("Maurice must implement the shared face extractor")


def extract_voice_features(path):
    """Kudakwashe: return one 2-D row using the training audio extractor.

    Document resampling, mono conversion, silence handling and feature order.
    Use pretrained speaker embeddings for verification, with MFCCs for analysis.
    """
    raise NotImplementedError("Kudakwashe must implement the shared voice extractor")
