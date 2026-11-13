"""CPU YuNet detection and aligned SFace embeddings (OpenCV Zoo)."""
from pathlib import Path
import hashlib
import os
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageOps

WEIGHTS = {
    'detector': ('face_detection_yunet_2023mar.onnx',
                 'face_detection_yunet/face_detection_yunet_2023mar.onnx'),
    'recognizer': ('face_recognition_sface_2021dec.onnx',
                   'face_recognition_sface/face_recognition_sface_2021dec.onnx'),
}
BASE_URL = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/'


class FaceInputError(ValueError):
    """Unreadable image or an image without exactly one sufficiently clear face."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(directory, download=False):
    """Allow offline operation; downloads are explicit and written atomically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, remote) in WEIGHTS.items():
        target = directory / name
        if not target.exists():
            if not download:
                raise FileNotFoundError(f'Missing {target}. Enable DOWNLOAD_WEIGHTS or copy ONNX files here.')
            descriptor, temporary = tempfile.mkstemp(dir=directory, suffix='.download')
            os.close(descriptor)
            try:
                with urllib.request.urlopen(BASE_URL + remote, timeout=120) as response:
                    with open(temporary, 'wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if Path(temporary).stat().st_size < 10000:
                    raise ValueError('Download is too small to be a model; check the URL/network')
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        paths[key] = target
    return paths


def read_rgb(path):
    try:
        with Image.open(path) as source:
            return np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy()
    except (OSError, ValueError) as error:
        raise FaceInputError(f'Cannot decode {path}: {error}') from error


class FaceEncoder:
    """Detect exactly one face, align landmarks, embed, then L2 normalize."""
    def __init__(self, weights_dir, detection_threshold=.8, max_side=1200):
        import cv2
        self.cv = cv2
        self.paths = ensure_weights(weights_dir)
        self.detection_threshold = detection_threshold
        self.max_side = max_side
        self.detector = cv2.FaceDetectorYN.create(
            str(self.paths['detector']), '', (320, 320), detection_threshold, .3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(self.paths['recognizer']), '')

    def align(self, rgb):
        cv = self.cv
        bgr = cv.cvtColor(np.asarray(rgb, dtype=np.uint8), cv.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        ratio = min(1., self.max_side / max(height, width))
        if ratio < 1:
            bgr = cv.resize(bgr, (round(width * ratio), round(height * ratio)))
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        try:
            _, faces = self.detector.detect(bgr)
        except cv.error as error:
            raise FaceInputError(f'Face detection failed: {error}') from error
        if faces is None or len(faces) != 1:
            count = 0 if faces is None else len(faces)
            raise FaceInputError(f'Expected exactly one face, detected {count}')
        if min(faces[0, 2:4]) < 30:
            raise FaceInputError('Face is too small; provide a closer, sharper photograph')
        return self.recognizer.alignCrop(bgr, faces[0])

    def embed_aligned(self, aligned_bgr):
        vector = self.recognizer.feature(np.ascontiguousarray(aligned_bgr)).reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm < 1e-10:
            raise FaceInputError('Encoder produced an invalid embedding')
        return (vector / norm).astype(np.float32)

    def extract(self, path):
        return self.embed_aligned(self.align(read_rgb(path))).reshape(1, -1)
