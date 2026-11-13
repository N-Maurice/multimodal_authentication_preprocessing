"""Face-only demonstration: python -m face_auth --image photo.jpg."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import joblib
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from face_auth.embeddings import FaceInputError
    from feature_extraction import extract_face_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/face_model.joblib')
    parser.add_argument('--weights', type=Path, default=Path(os.environ.get('FACE_WEIGHTS_DIR', ROOT / 'models/pretrained')))
    args = parser.parse_args()
    try:
        model = joblib.load(args.model)
        features = extract_face_features(args.image, weights_dir=args.weights)
        identity = model.predict(features)[0]
        if identity == 'unknown':
            parser.exit(1, 'Access denied: face was not verified\n')
        print(f'Face verified: {identity}. Voice confirmation is still required.')
    except FaceInputError as error:
        parser.exit(1, f'Access denied: {error}\n')
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f'Setup error: {error}\n')


if __name__ == '__main__':
    main()
