"""Source isolation and rejection behavior; no downloaded weights required."""
import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from face_auth.pipeline import (FaceAuthenticator, calibrate_similarity,
                                image_variants, inventory, split_sources)


def source_frame():
    return pd.DataFrame([
        {'user_id': user, 'source_id': f'{user}-{i}', 'status': 'ok'}
        for user, count in [('a', 4), ('b', 3), ('c', 1), ('unauthorized', 3)]
        for i in range(count)
    ])


def test_splits_keep_underrepresented_and_explicit_unknowns_out_of_training():
    frame = split_sources(source_frame())
    assert set(frame[frame.split == 'train'].user_id) == {'a', 'b'}
    assert set(frame[frame.user_id.isin(['c', 'unauthorized'])].split) == {'unknown_test'}
    assert frame.groupby('source_id').split.nunique().max() == 1
    pd.testing.assert_frame_equal(frame, split_sources(source_frame()))


def test_split_rejects_insufficient_enrollment():
    with pytest.raises(ValueError, match='two identities'):
        split_sources(source_frame().query("user_id != 'b'"))


def test_inventory_excludes_duplicates_and_reports_corruption(tmp_path):
    folder = tmp_path / 'a'
    folder.mkdir()
    for name in ['one.png', 'copy.png']:
        Image.new('RGB', (40, 40), 'red').save(folder / name)
    (folder / 'broken.jpg').write_bytes(b'not an image')
    frame = inventory(tmp_path)
    assert frame.status.value_counts().to_dict() == {'ok': 1, 'duplicate': 1, 'decode_error': 1}


def test_conflicting_duplicate_labels_are_rejected(tmp_path):
    for user in ['a', 'b']:
        folder = tmp_path / user
        folder.mkdir()
        Image.new('RGB', (40, 40), 'red').save(folder / 'one.png')
    with pytest.raises(ValueError, match='conflicting'):
        inventory(tmp_path)


def trained():
    X = np.array([[1., 0., 0.], [.99, .01, 0.], [0., 1., 0.], [.01, .99, 0.]])
    return FaceAuthenticator(C=10, similarity_threshold=.8, confidence_threshold=.5).fit(X, ['a', 'a', 'b', 'b'])


def test_distance_gate_rejects_unknown_even_with_permissive_confidence():
    model = trained()
    assert model.predict([[1, 0, 0]])[0] == 'a'
    assert model.predict([[0, 1, 0]])[0] == 'b'
    assert model.predict([[0, 0, 1]])[0] == 'unknown'
    assert np.array_equal(model.predict_proba([[0, 0, 1]]), [[0, 0]])


def test_reload_parity_and_validation_calibration(tmp_path):
    model = trained()
    curve = calibrate_similarity(model, [[1, 0, 0], [0, 1, 0]], ['a', 'b'])
    assert .363 <= model.similarity_threshold <= .95
    assert curve.balanced_score.max() == 1.
    joblib.dump(model, tmp_path / 'face.joblib')
    reloaded = joblib.load(tmp_path / 'face.joblib')
    X = [[1, 0, 0], [0, 0, 1]]
    np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))


@pytest.mark.parametrize('X', [[[float('nan'), 0, 0]], [[0, 0, 0]], [[1, 0]]])
def test_invalid_embeddings_fail_closed(X):
    with pytest.raises(ValueError):
        trained().predict(X)


def test_filters_preserve_rgb_shape_and_original():
    rgb = np.full((112, 112, 3), 128, dtype=np.uint8)
    variants = image_variants(rgb)
    assert len(variants) >= 3
    np.testing.assert_array_equal(variants['original'], rgb)
    for image in variants.values():
        assert np.asarray(image).shape == rgb.shape
