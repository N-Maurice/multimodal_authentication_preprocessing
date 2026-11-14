"""Sequential face -> voice -> product demonstration using trusted local artifacts.

Flow (matches the assignment diagram):
  1. FACE    a face image is checked first. If it is not a registered user, stop here.
  2. REQUEST the recognised user asks for a product prediction.
  3. VOICE   a voice sample must confirm the SAME user. Only then is the product model run.

Every attempt (granted or denied) is appended to reports/access_log.csv.
Exit codes: 0 = access granted, 1 = access denied, 2 = setup/usage error.

Examples
  python main.py --face data/raw/images/larissa/Larissa_Surprised.jpeg \\
                 --voice "data/raw/audio/larissa/Larissa_Confirm transaction.ogg"
  python main.py            # prompts for the face and voice file paths
"""
import argparse
import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # lets joblib find face_auth.pipeline.FaceAuthenticator
    sys.path.insert(0, str(ROOT))

# Owner hints shown when an artifact is missing.
ARTIFACT_HINTS = {
    "face_model.joblib": "Maurice: run notebooks/02_face_recognition.ipynb and commit "
                         "models/face_model.joblib (it is listed in .gitignore, use `git add -f`)",
    "voice_model.joblib": "Kudakwashe: run notebooks/03_voice_verification.ipynb",
    "product_model.joblib": "Patrick: run notebooks/01_product_recommendation.ipynb "
                            "(or python rebuild_product_model.py)",
    "customer_inference.csv": "run python rebuild_product_model.py (or the last cell of notebook 01)",
    "user_registry.csv": "fill data/processed/user_registry.csv",
}


class AccessDenied(Exception):
    """An authentication checkpoint rejected the attempt.

    stage is 'face', 'voice' or 'match' so the caller can report where it stopped.
    """

    def __init__(self, message, stage="auth"):
        super().__init__(message)
        self.stage = stage


# --------------------------------------------------------------------------- checks
def identify(model, features, threshold, stage="auth"):
    """Return (identity, confidence). Models must expose predict_proba and classes_.

    The face model (face_auth.pipeline.FaceAuthenticator) already applies its own
    similarity gate and returns all-zero scores for a rejected face, so a rejected face
    has confidence 0 and falls below any positive threshold.
    """
    import numpy as np

    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or not np.isfinite(probabilities).all():
        raise AccessDenied("Invalid authentication scores", stage)
    index = int(probabilities[0].argmax())
    return str(model.classes_[index]), float(probabilities[0, index])


def verify_face(face_model, features, registry, threshold):
    identity, confidence = identify(face_model, features, threshold, "face")
    if confidence < threshold or confidence <= 0 or identity not in registry:
        raise AccessDenied("Face was not recognised as a registered user", "face")
    return identity, confidence


def verify_voice(voice_model, features, registry, threshold, expected_identity):
    identity, confidence = identify(voice_model, features, threshold, "voice")
    if confidence < threshold or identity not in registry:
        raise AccessDenied("Voice was not verified as a registered user", "voice")
    if identity != expected_identity:
        raise AccessDenied("Voice belongs to a different user than the face", "match")
    return identity, confidence


def authenticate(face_model, voice_model, face_features, voice_features, registry,
                 face_threshold, voice_threshold):
    """Face first, then voice of the same user. Unknowns are rejected."""
    identity, _ = verify_face(face_model, face_features, registry, face_threshold)
    verify_voice(voice_model, voice_features, registry, voice_threshold, identity)
    return identity


# --------------------------------------------------------------------------- setup
def _load_model(path, label):
    """joblib.load with clear errors for missing files and scikit-learn version mismatches."""
    import joblib

    if not path.exists():
        hint = ARTIFACT_HINTS.get(path.name, "")
        raise ValueError(f"Missing {label} file {path}" + (f" ({hint})" if hint else ""))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except Exception as error:  # unpickling failures are not one exception type
            raise ValueError(
                f"Could not load {path.name}: {type(error).__name__}: {error}. It was probably "
                "saved with a different scikit-learn version. Everyone must use the same version "
                "(check: python -c \"import sklearn; print(sklearn.__version__)\") and re-save "
                "the model from its notebook.") from error
    for item in caught:
        if "InconsistentVersion" in item.category.__name__:
            print(f"Note: {path.name} was saved with a different scikit-learn version; "
                  "results may differ.", file=sys.stderr)
            break
    return model


def load_context(args):
    """Load and validate every trusted local artifact. Raises ValueError/OSError on problems."""
    import pandas as pd

    # Imported here so unit tests and --help work without the heavy audio/face dependencies.
    from feature_extraction import extract_face_features, extract_voice_features

    metadata = json.loads((args.models / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "trained":
        raise ValueError('models/model_metadata.json must have "status": "trained"')
    for part in ("face", "voice"):
        if metadata[part].get("threshold") is None:
            raise ValueError(f"metadata.{part}.threshold is not set")
    if not metadata["product"].get("feature_columns"):
        raise ValueError("metadata.product.feature_columns is empty")

    for path in (args.registry, args.customers):
        if not path.exists():
            raise ValueError(f"Missing {path} ({ARTIFACT_HINTS.get(path.name, '')})")
    registry_frame = pd.read_csv(args.registry, dtype=str)
    if registry_frame.empty:
        raise ValueError("The user registry is empty")
    if registry_frame.user_id.duplicated().any():
        raise ValueError("Registry user_id must be unique")
    customers = pd.read_csv(args.customers, dtype={"customer_id": str})

    return {
        "metadata": metadata,
        "registry": registry_frame.set_index("user_id").customer_id.to_dict(),
        "names": registry_frame.set_index("user_id").display_name.to_dict(),
        "customers": customers,
        # The face model applies its own cosine-similarity gate (face.threshold) internally;
        # confidence_threshold is the minimum classifier probability it also requires.
        "face_gate": float(metadata["face"].get("confidence_threshold", 0.5)),
        "face_model": _load_model(args.models / "face_model.joblib", "face model"),
        "voice_model": _load_model(args.models / "voice_model.joblib", "voice model"),
        "product_model": _load_model(args.models / "product_model.joblib", "product model"),
        "extract_face": extract_face_features,
        "extract_voice": extract_voice_features,
    }


# --------------------------------------------------------------------------- one attempt
def attempt(ctx, face_path, voice_path, say=print, pause=None):
    """Run one full transaction attempt and return a result dict (never raises AccessDenied).

    result = {outcome: 'granted'|'denied', stage, identity, confidence_face,
              confidence_voice, prediction, product_confidence, reason}
    The product model is only called after BOTH checks pass. An unreadable image or audio
    file (no face, several faces, silence, wrong format) is a denial, not a crash.
    """
    meta, registry = ctx["metadata"], ctx["registry"]
    result = {"outcome": "denied", "stage": "face", "identity": "", "confidence_face": "",
              "confidence_voice": "", "prediction": "", "product_confidence": "", "reason": ""}
    try:
        # 1. FACE
        say("[1/3] Checking face ...")
        try:
            face_features = ctx["extract_face"](face_path)
        except ValueError as error:  # includes FaceInputError: no/multiple/small face, bad file
            raise AccessDenied(f"Face image could not be used ({error})", "face")
        identity, face_conf = verify_face(ctx["face_model"], face_features, registry,
                                          ctx.get("face_gate", 0.5))
        result.update(identity=identity, confidence_face=round(face_conf, 3), stage="request")
        name = ctx["names"].get(identity, identity)
        say(f"      Face recognised: {name} (confidence {face_conf:.2f}). "
            "Prediction access unlocked pending voice confirmation.")

        # 2. REQUEST
        if pause:
            pause(name)
        say(f"[2/3] {name} requested a product prediction. Voice confirmation required.")

        # 3. VOICE (extractor only runs now, after the face has passed)
        result["stage"] = "voice"
        try:
            voice_features = ctx["extract_voice"](voice_path)
        except ValueError as error:  # undecodable audio or no speech after silence trimming
            raise AccessDenied(f"Voice sample could not be used ({error})", "voice")
        _, voice_conf = verify_voice(ctx["voice_model"], voice_features, registry,
                                     meta["voice"]["threshold"], identity)
        result["confidence_voice"] = round(voice_conf, 3)
        say(f"[3/3] Voice approved: matches {name} (confidence {voice_conf:.2f}).")

        # PRODUCT MODEL (only reachable when both checks passed)
        rows = ctx["customers"][ctx["customers"].customer_id == registry[identity]]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one inference row for customer {registry[identity]}")
        columns = meta["product"]["feature_columns"]
        model = ctx["product_model"]
        result["prediction"] = str(model.predict(rows[columns])[0])
        if hasattr(model, "predict_proba"):
            result["product_confidence"] = round(float(model.predict_proba(rows[columns]).max()), 3)
        result.update(outcome="granted", stage="done")
    except AccessDenied as error:
        result.update(outcome="denied", stage=error.stage, reason=str(error))
    return result


def log_attempt(path, face_path, voice_path, result):
    """Append the attempt to a CSV log. Predictions are never logged for denied attempts."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["time", "face_file", "voice_file", "outcome", "stopped_at",
                                 "identity", "reason"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), Path(face_path).name,
                             Path(voice_path).name if voice_path else "", result["outcome"],
                             result["stage"], result["identity"], result["reason"]])
    except OSError as error:
        print(f"(could not write access log: {error})")


# --------------------------------------------------------------------------- CLI
def _ask(prompt):
    return input(prompt).strip().strip('"').strip("'")


def run(args):
    face_path = args.face or Path(_ask("Path to face image: "))
    if not face_path.exists():
        raise ValueError(f"Face file not found: {face_path}")
    ctx = load_context(args)

    # Ask for the voice file only after the face passes, as in the real flow.
    voice_path = {"value": args.voice}

    def pause(name):
        if voice_path["value"] is None:
            voice_path["value"] = Path(_ask(f"{name}, path to your voice sample: "))
        elif not args.yes:
            input(f"Press Enter to request a prediction as {name} ... ")

    original = ctx["extract_voice"]

    def extract_voice(_ignored):
        if not voice_path["value"].exists():
            raise ValueError(f"Voice file not found: {voice_path['value']}")
        return original(voice_path["value"])

    ctx["extract_voice"] = extract_voice

    print("=== Multimodal product recommendation ===")
    result = attempt(ctx, face_path, None, pause=pause)
    log_attempt(args.log, face_path, voice_path["value"], result)
    if result["outcome"] == "granted":
        confidence = f" ({result['product_confidence']:.0%} model confidence)" \
            if result["product_confidence"] != "" else ""
        print(f"ACCESS GRANTED | Verified user: {result['identity']} | "
              f"Recommended product: {result['prediction']}{confidence}")
        return 0
    label = {"match": "identity-match"}.get(result["stage"], result["stage"])
    print(f"ACCESS DENIED at the {label} check: {result['reason']}. No prediction was made.")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--face", type=Path, help="face image (asked for if omitted)")
    parser.add_argument("--voice", type=Path, help="voice sample (asked for after the face passes)")
    parser.add_argument("--yes", action="store_true", help="skip the 'press Enter' request step")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/processed/user_registry.csv")
    parser.add_argument("--customers", type=Path, default=ROOT / "data/processed/customer_inference.csv")
    parser.add_argument("--log", type=Path, default=ROOT / "reports/access_log.csv")
    args = parser.parse_args()
    try:
        parser.exit(run(args))
    except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as error:
        parser.exit(2, f"Setup error: {error}\n")


if __name__ == "__main__":
    main()
