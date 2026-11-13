"""Sequential face -> voice -> product demonstration using trusted local artifacts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class AccessDenied(Exception):
    """An authentication checkpoint rejected the attempt."""


def authenticate(face_model, voice_model, face_features, voice_features, registry,
                 face_threshold, voice_threshold):
    """Models must expose predict_proba and classes_; unknowns must be rejected."""
    def identify(model, features, threshold):
        import numpy as np
        probabilities = model.predict_proba(features)
        if probabilities.shape[0] != 1 or not np.isfinite(probabilities).all():
            raise AccessDenied("Invalid authentication scores")
        index = int(probabilities[0].argmax())
        identity = str(model.classes_[index])
        if float(probabilities[0, index]) < threshold or identity not in registry:
            raise AccessDenied("Identity was not verified")
        return identity

    identity = identify(face_model, face_features, face_threshold)
    voice_identity = identify(voice_model, voice_features, voice_threshold)
    if identity != voice_identity:
        raise AccessDenied("Voice identity does not match face identity")
    return identity


def run(args):
    import joblib
    import pandas as pd

    # Import only after setup: each notebook must implement these shared extractors.
    from feature_extraction import extract_face_features, extract_voice_features

    metadata = json.loads((args.models / "model_metadata.json").read_text())
    if metadata.get("status") != "trained":
        raise ValueError("Complete the training notebooks and mark metadata as trained first")
    registry_frame = pd.read_csv(args.registry, dtype=str)
    if registry_frame.user_id.duplicated().any():
        raise ValueError("Registry user_id must be unique")
    registry = registry_frame.set_index("user_id").customer_id.to_dict()
    face_model = joblib.load(args.models / "face_model.joblib")
    voice_model = joblib.load(args.models / "voice_model.joblib")
    identity = authenticate(
        face_model, voice_model,
        extract_face_features(args.face), extract_voice_features(args.voice), registry,
        metadata["face"]["threshold"], metadata["voice"]["threshold"],
    )
    customers = pd.read_csv(args.customers, dtype={"customer_id": str})
    customer = customers[customers.customer_id == registry[identity]]
    if len(customer) != 1:
        raise ValueError("Expected exactly one inference row for the verified customer")
    product_model = joblib.load(args.models / "product_model.joblib")
    prediction = product_model.predict(customer[metadata["product"]["feature_columns"]])[0]
    print(f"Verified user: {identity} | Recommended product: {prediction}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/processed/user_registry.csv")
    parser.add_argument("--customers", type=Path, default=ROOT / "data/processed/customer_inference.csv")
    args = parser.parse_args()
    try:
        run(args)
    except AccessDenied as error:
        parser.exit(1, f"Access denied: {error}\n")
    except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as error:
        parser.exit(2, f"Setup error: {error}\n")


if __name__ == "__main__":
    main()
