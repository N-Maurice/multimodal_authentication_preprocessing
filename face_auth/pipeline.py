"""Auditable source splits, training-only augmentation and unknown rejection."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import normalize

from .embeddings import FaceInputError, read_rgb

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
UNKNOWN_FOLDERS = {'unauthorized', 'unknown'}


def inventory(directory):
    rows = []
    for path in sorted(Path(directory).rglob('*')):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        relative = path.relative_to(directory)
        if len(relative.parts) != 2:
            raise ValueError(f'Use images/<user_id>/<filename>: {path}')
        user_id = relative.parts[0].lower()
        try:
            rgb = read_rgb(path)
            gray = np.asarray(Image.fromarray(rgb).convert('L'), dtype=float)
            # Pixel hash also catches copies with altered EXIF metadata.
            digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            rows.append(dict(path=str(path), user_id=user_id, source_id=digest,
                             width=rgb.shape[1], height=rgb.shape[0],
                             brightness=float(gray.mean()), contrast=float(gray.std()),
                             status='ok', error=''))
        except FaceInputError as error:
            rows.append(dict(path=str(path), user_id=user_id, source_id=None,
                             status='decode_error', error=str(error)))
    if not rows:
        raise ValueError(f'No supported images found in {directory}')
    frame = pd.DataFrame(rows)
    valid = frame[frame.status == 'ok']
    conflicts = valid.groupby('source_id').user_id.nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical images have conflicting user IDs; resolve them before splitting')
    duplicates = frame.status.eq('ok') & frame.source_id.duplicated(keep='first')
    frame.loc[duplicates, 'status'] = 'duplicate'
    return frame


def split_sources(frame, seed=42):
    """At least three originals per enrolled identity: train, validation, test.

    Identities with fewer originals are withheld entirely as unknown test users.
    Near duplicates and captures from one session still require manual review.
    """
    frame = frame.copy()
    frame['split'] = 'excluded'
    rng = np.random.default_rng(seed)
    for user_id, group in frame[frame.status == 'ok'].groupby('user_id', sort=True):
        indices = rng.permutation(group.index.to_numpy())
        if user_id in UNKNOWN_FOLDERS or len(indices) < 3:
            frame.loc[indices, 'split'] = 'unknown_test'
        else:
            frame.loc[indices[:-2], 'split'] = 'train'
            frame.loc[indices[-2], 'split'] = 'validation'
            frame.loc[indices[-1], 'split'] = 'test'
    if frame.loc[frame.split == 'train', 'user_id'].nunique() < 2:
        raise ValueError('Need at least two identities with three independent, detectable images each')
    assert frame[frame.split != 'excluded'].groupby('source_id').split.nunique().max() == 1
    return frame


def image_variants(rgb):
    """Deterministic mild augmentation and filters for inspection/training."""
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    image = Image.fromarray(rgb)
    # CLAHE affects LAB lightness only, retaining the original chromatic channels.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    clahe_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Explicit convention: output = 255 * (input / 255) ** gamma.
    # Gamma 0.8 brightens midtones; one deterministic setting is one new variant.
    gamma_lut = np.clip(np.round(255 * (np.arange(256) / 255.) ** .8), 0, 255).astype(np.uint8)
    return {
        'original': image,
        'rotation_plus_8': image.rotate(8, resample=Image.Resampling.BICUBIC),
        'rotation_minus_8': image.rotate(-8, resample=Image.Resampling.BICUBIC),
        'horizontal_flip': ImageOps.mirror(image),
        'grayscale': ImageOps.grayscale(image).convert('RGB'),
        'brightness_090': ImageEnhance.Brightness(image).enhance(.9),
        'contrast_110': ImageEnhance.Contrast(image).enhance(1.1),
        'gaussian_blur': image.filter(ImageFilter.GaussianBlur(.6)),
        'sharpen': image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)),
        'clahe': Image.fromarray(clahe_rgb),
        'gamma_080': Image.fromarray(cv2.LUT(rgb, gamma_lut)),
        'median_blur_3': Image.fromarray(cv2.medianBlur(rgb, 3)),
    }


def detect_inventory(frame, encoder):
    """Audit unusable faces before assigning partitions; never silently drop them."""
    frame = frame.copy()
    aligned = {}
    for index, row in frame[frame.status == 'ok'].iterrows():
        try:
            aligned[row.source_id] = encoder.align(read_rgb(row.path))
        except FaceInputError as error:
            frame.loc[index, ['status', 'error']] = ['face_error', str(error)]
    return frame, aligned


def build_features(manifest, aligned, encoder):
    rows = []
    for row in manifest[manifest.split != 'excluded'].itertuples():
        crop = aligned[row.source_id]
        # Reuse landmark-aligned crops. Do not detect landmarks again after filters.
        rgb = encoder.cv.cvtColor(crop, encoder.cv.COLOR_BGR2RGB)
        variants = image_variants(rgb) if row.split == 'train' else {'original': Image.fromarray(rgb)}
        for name, image in variants.items():
            bgr = encoder.cv.cvtColor(np.array(image), encoder.cv.COLOR_RGB2BGR)
            feature = encoder.embed_aligned(bgr)
            record = dict(user_id=row.user_id, source_id=row.source_id, split=row.split,
                          augmentation=name, path=row.path)
            record.update({f'embedding_{i:03d}': float(value) for i, value in enumerate(feature)})
            rows.append(record)
    frame = pd.DataFrame(rows)
    if (frame[frame.split != 'train'].augmentation != 'original').any():
        raise AssertionError('Validation/test augmentation is forbidden')
    if frame.groupby('source_id').split.nunique().max() != 1:
        raise AssertionError('Source leakage between splits')
    return frame


class FaceAuthenticator(ClassifierMixin, BaseEstimator):
    """Identity classifier with a class-specific cosine similarity rejection gate.

    predict_proba returns all zeros for rejected samples: these are intentional
    rejection scores, NOT normalized probability distributions. Use classifier_
    directly for closed-set log loss. The CLI rejects zero scores.
    """
    def __init__(self, C=1., similarity_threshold=.363, confidence_threshold=.5):
        self.C = C
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

    def fit(self, X, y):
        X = normalize(np.asarray(X))
        y = np.asarray(y)
        self.classifier_ = LogisticRegression(C=self.C, max_iter=2000, random_state=42).fit(X, y)
        self.classes_ = self.classifier_.classes_
        self.centroids_ = normalize(np.array([X[y == label].mean(axis=0) for label in self.classes_]))
        self.n_features_in_ = X.shape[1]
        return self

    def scores(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('Face features must be a finite 2-D array with the trained feature width')
        if (np.linalg.norm(X, axis=1) < 1e-10).any():
            raise ValueError('Zero embedding is invalid')
        X = normalize(X)
        return self.classifier_.predict_proba(X), X @ self.centroids_.T

    def predict_proba(self, X):
        probabilities, similarities = self.scores(X)
        indices = probabilities.argmax(axis=1)
        similarities = similarities[np.arange(len(indices)), indices]
        accepted = ((similarities >= self.similarity_threshold) &
                    (probabilities.max(axis=1) >= self.confidence_threshold))
        return probabilities * accepted[:, None]

    def predict(self, X):
        scores = self.predict_proba(X)
        return np.where(scores.max(axis=1) > 0, self.classes_[scores.argmax(axis=1)], 'unknown')


def closed_metrics(model, X, y):
    probabilities, _ = model.scores(X)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    return dict(accuracy=float(accuracy_score(y, predictions)),
                macro_f1=float(f1_score(y, predictions, labels=model.classes_, average='macro', zero_division=0)),
                log_loss=float(log_loss(y, probabilities, labels=model.classes_)))


def calibrate_similarity(model, X_validation, y_validation):
    """Genuine vs wrong-identity validation comparisons; not unknown-user FAR.

    The published SFace .363 cosine threshold is a conservative lower bound.
    Select once on validation comparisons. Never read test samples here.
    """
    _, similarities = model.scores(X_validation)
    y_validation = np.asarray(y_validation)
    correct = y_validation[:, None] == model.classes_[None, :]
    genuine, impostor = similarities[correct], similarities[~correct]
    rows = []
    for threshold in np.linspace(.363, .95, 120):
        tpr = float(np.mean(genuine >= threshold))
        proxy_far = float(np.mean(impostor >= threshold))
        rows.append(dict(threshold=float(threshold), genuine_acceptance=tpr,
                         wrong_identity_acceptance=proxy_far,
                         balanced_score=(tpr + 1 - proxy_far) / 2))
    curve = pd.DataFrame(rows)
    # Prefer the lower threshold among equally good validation scores.
    best = curve.sort_values(['balanced_score', 'threshold'], ascending=[False, True]).iloc[0]
    model.similarity_threshold = float(best.threshold)
    return curve
