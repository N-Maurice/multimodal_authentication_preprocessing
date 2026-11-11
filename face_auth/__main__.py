"""Face-only demonstration: python -m face_auth --image photo.jpg."""
import argparse
import os
from pathlib import Path


def main():
    import joblib
    from .embeddings import FaceEncoder, FaceInputError

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=root / 'models/face_model.joblib')
    parser.add_argument('--weights', type=Path, default=Path(os.environ.get('FACE_WEIGHTS_DIR', root / 'models/pretrained')))
    args = parser.parse_args()
    try:
        model = joblib.load(args.model)
        features = FaceEncoder(args.weights).extract(args.image)
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
