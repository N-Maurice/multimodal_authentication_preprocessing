"""Source isolation and rejection behavior; no downloaded weights required."""
from pathlib import Path

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


def test_expression_split_gives_each_person_one_photo_per_split():
    from face_auth.pipeline import expression_from_filename, split_by_expression
    assert expression_from_filename('x/Larissa_Smilling.jpeg') == 'smiling'
    assert expression_from_filename('Maurice_surprised.JPG') == 'surprised'
    rows = [dict(user_id=user, path=f'{user}/{user}_{expr}.jpg', source_id=f'{user}-{expr}', status='ok')
            for user in ['a', 'b'] for expr in ['Neutral', 'Smiling', 'Surprised']]
    rows.append(dict(user_id='c', path='c/c_Neutral.jpg', source_id='c-n', status='ok'))
    frame = split_by_expression(pd.DataFrame(rows))
    enrolled = frame[frame.user_id.isin(['a', 'b'])]
    assert (enrolled.groupby('user_id').split.value_counts().unstack() == 1).all().all()
    assert set(enrolled[enrolled.split == 'train'].expression) == {'neutral'}
    assert set(enrolled[enrolled.split == 'test'].expression) == {'surprised'}
    assert frame.loc[frame.user_id == 'c', 'split'].tolist() == ['unknown_test']
    with pytest.raises(ValueError):
        split_by_expression(pd.DataFrame(rows), {'neutral': 'train', 'smiling': 'train', 'surprised': 'test'})


def test_calibration_takes_middle_of_tied_thresholds():
    model = trained()
    curve = calibrate_similarity(model, [[1, 0, 0], [0, 1, 0]], ['a', 'b'])
    tied = curve.loc[curve.balanced_score == curve.balanced_score.max(), 'threshold']
    assert tied.min() < model.similarity_threshold < tied.max()


def test_leave_one_identity_out_never_trains_on_hidden_person():
    from face_auth.pipeline import leave_one_identity_out
    rows = []
    for index, user in enumerate(['a', 'b', 'c']):
        for split in ['train', 'validation', 'test']:
            vector = np.zeros(3)
            vector[index] = 1
            rows.append(dict(user_id=user, source_id=f'{user}-{split}', split=split,
                             augmentation='original', path=f'{user}-{split}.png',
                             e0=vector[0], e1=vector[1], e2=vector[2]))
    results = leave_one_identity_out(pd.DataFrame(rows), ['e0', 'e1', 'e2'], C=10)
    impostors = results[results.cohort == 'impostor']
    assert set(impostors.hidden_identity) == {'a', 'b', 'c'}
    assert (impostors.user_id == impostors.hidden_identity).all()
    assert len(impostors) == 9
    # Orthogonal embeddings: every hidden person must be rejected.
    assert (impostors.prediction == 'unknown').all()
    genuine = results[results.cohort == 'genuine']
    assert (genuine.prediction == genuine.user_id).all()


@pytest.mark.parametrize('X', [[[float('nan'), 0, 0]], [[0, 0, 0]], [[1, 0]]])
def test_invalid_embeddings_fail_closed(X):
    with pytest.raises(ValueError):
        trained().predict(X)


def test_filters_preserve_rgb_shape_and_original():
    rgb = np.full((112, 112, 3), 128, dtype=np.uint8)
    variants = image_variants(rgb)
    assert len(variants) == 13
    assert {'rotation_plus_8', 'horizontal_flip', 'grayscale', 'affine', 'brightness_090', 'gaussian_blur'} <= variants.keys()
    assert {'clahe', 'gamma_080', 'median_blur_3'} <= variants.keys()
    np.testing.assert_array_equal(variants['original'], rgb)
    for image in variants.values():
        assert np.asarray(image).shape == rgb.shape


def test_processed_export_writes_twenty_one_named_images_and_preserves_splits(tmp_path):
    from face_auth.preparation import export_processed_images
    raw = tmp_path / 'raw'
    for user_index, (user, count) in enumerate([('maurice', 3), ('larissa', 3), ('kudakwashe', 1)]):
        folder = raw / user
        folder.mkdir(parents=True)
        for number in range(count):
            # Each source has distinct pixels so inventory does not collapse it.
            Image.new('RGB', (40, 40), (number * 60, user_index * 60, 30)).save(folder / f'Expression {number}.png')
    manifest = split_sources(inventory(raw))
    output = tmp_path / 'processed/images'
    exported, skipped = export_processed_images(manifest, output)
    assert exported.groupby('user_id').size().to_dict() == {'larissa': 21, 'maurice': 21}
    assert skipped.user_id.tolist() == ['kudakwashe']
    assert exported.groupby('source_id').size().eq(7).all()
    assert exported.groupby('source_id').split.nunique().eq(1).all()
    for row in exported.itertuples():
        path = Path(row.output_path)
        assert path.is_file()
        assert path.name.startswith(row.user_id + '__source_')
        assert path.name.endswith('__' + row.augmentation + '.png')
        assert ' ' not in path.name
    assert (output.parent / 'processed_image_manifest.csv').is_file()
    # Reruns cannot multiply the prepared dataset.
    export_processed_images(manifest, output)
    assert len(list(output.rglob('*.png'))) == 42


def test_each_notebook_code_cell_has_preceding_markdown():
    import json
    from pathlib import Path
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/02_face_recognition.ipynb'
    cells = json.loads(notebook_path.read_text())['cells']
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            assert index > 0 and cells[index - 1]['cell_type'] == 'markdown'


def test_new_filters_have_expected_pixel_effects():
    rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
    rgb[16, 16] = 255
    variants = image_variants(rgb)
    assert np.asarray(variants['gamma_080'])[0, 0, 0] > rgb[0, 0, 0]
    assert np.asarray(variants['median_blur_3'])[16, 16, 0] == 100
    assert not np.array_equal(np.asarray(variants['clahe']), rgb)
    for name in ['clahe', 'gamma_080', 'median_blur_3']:
        assert np.asarray(variants[name]).dtype == np.uint8


def test_affine_variant_changes_pixels_but_keeps_shape():
    rgb = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    affine = np.asarray(image_variants(rgb)['affine'])
    assert affine.shape == rgb.shape and affine.dtype == np.uint8
    assert not np.array_equal(affine, rgb)


def test_build_features_keeps_augmented_copies_in_source_split():
    from face_auth.pipeline import build_features

    class FakeEncoder:
        import cv2 as cv

        def embed_aligned(self, bgr):
            vector = np.asarray(bgr, dtype=float).mean(axis=(0, 1)) + 1
            return vector / np.linalg.norm(vector)

    manifest = pd.DataFrame([dict(user_id='a', source_id=f's{i}', split=split, path=f'{i}.png')
                             for i, split in enumerate(['train', 'validation', 'test'])])
    crops = {f's{i}': np.full((112, 112, 3), 40 * (i + 1), dtype=np.uint8) for i in range(3)}
    frame = build_features(manifest, crops, FakeEncoder(), augment_splits=('train', 'validation', 'test'))
    assert frame.groupby('source_id').size().eq(13).all()
    assert (frame.groupby('source_id').split.nunique() == 1).all()
    assert dict(zip(frame.source_id, frame.split)) == {'s0': 'train', 's1': 'validation', 's2': 'test'}
    only_train = build_features(manifest, crops, FakeEncoder())
    assert (only_train[only_train.split != 'train'].augmentation == 'original').all()


def test_resize_photos_limits_longest_side_and_keeps_raw(tmp_path):
    from face_auth.preparation import resize_photos
    raw = tmp_path / 'raw' / 'patrick'
    raw.mkdir(parents=True)
    Image.new('RGB', (3024, 1900), 'blue').save(raw / 'Patrick_Neutral.jpeg')
    Image.new('RGB', (600, 800), 'red').save(raw / 'Patrick_Smiling.png')
    table = resize_photos(tmp_path / 'raw', tmp_path / 'resized', max_side=1024)
    sizes = dict(zip(table.raw_path.map(lambda p: Path(p).name), zip(table.width, table.height)))
    assert sizes == {'Patrick_Neutral.jpeg': (1024, 643), 'Patrick_Smiling.png': (600, 800)}
    assert Image.open(raw / 'Patrick_Neutral.jpeg').size == (3024, 1900)
    (tmp_path / 'resized' / 'patrick' / 'stale.jpg').write_bytes(b'x')
    with pytest.raises(ValueError, match='stale'):
        resize_photos(tmp_path / 'raw', tmp_path / 'resized', max_side=1024)


WEIGHTS = Path(__file__).resolve().parents[1] / 'models' / 'pretrained'
IMAGES = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'images'


@pytest.mark.skipif(not (WEIGHTS / 'face_recognition_sface_2021dec.onnx').is_file(),
                    reason='pretrained ONNX weights not downloaded')
def test_extract_face_features_returns_one_row_and_rejects_bad_inputs(tmp_path):
    from face_auth.embeddings import FaceInputError
    from feature_extraction import EMBEDDING_SIZE, extract_face_features
    photos = sorted(IMAGES.glob('*/*_Neutral.jpeg'))[:2]
    features = extract_face_features(photos[0], weights_dir=WEIGHTS)
    assert features.shape == (1, EMBEDDING_SIZE)
    assert np.isclose(np.linalg.norm(features), 1, atol=1e-5)
    Image.new('RGB', (300, 300), 'white').save(tmp_path / 'blank.png')
    with pytest.raises(FaceInputError, match='detected 0'):
        extract_face_features(tmp_path / 'blank.png', weights_dir=WEIGHTS)
    left, right = (Image.open(p).convert('RGB') for p in photos)
    left.thumbnail((500, 500))
    right.thumbnail((500, 500))
    pair = Image.new('RGB', (left.width + right.width, max(left.height, right.height)), 'white')
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width, 0))
    pair.save(tmp_path / 'pair.png')
    with pytest.raises(FaceInputError, match='detected 2'):
        extract_face_features(tmp_path / 'pair.png', weights_dir=WEIGHTS)
