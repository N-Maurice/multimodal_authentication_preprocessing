"""Multimodal authentication app: sign in, face, voice, then a product recommendation.

This is the project's main application.

Interactive (webcam and microphone):
    python app.py
    python app.py --photo face.jpg --audio voice.ogg   # same guided flow using files

Scripted single attempt (used by run_demo.py, the tests and quick demonstrations):
    python app.py --face data/raw/images/larissa/Larissa_Surprised.jpeg \\
                  --voice "data/raw/audio/larissa/Larissa_Confirm transaction.ogg"

Interactive flow
  1. SIGN IN   enter your username (or email, if the registry has an email column)
  2. FACE      a small camera window opens; click the picture, press Space or the
               button to take your photo. The face must belong to your account.
  3. REQUEST   confirm that you want a product recommendation
  4. VOICE     say the passphrase; the voice must belong to the same account
  5. PRODUCT   the recommendation is shown only after both checks pass

Each step offers to retry or quit. After --max-attempts failed tries at the same step
(default 3) the session ends. Without a desktop display (or with --no-window) the
camera takes the photo automatically once one face is steady.

Every attempt (granted or denied) is appended to reports/access_log.csv.
Exit codes: 0 = access granted, 1 = denied or cancelled, 2 = setup/usage error.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # lets joblib find face_auth.pipeline.FaceAuthenticator
    sys.path.insert(0, str(ROOT))
PASSPHRASES = ("Confirm transaction", "Yes, approve")

# Owner hints shown when an artifact is missing.
ARTIFACT_HINTS = {
    "face_model.joblib": "Maurice: run notebooks/02_face_recognition.ipynb",
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


# --------------------------------------------------------------------------- terminal style
class Style:
    """ANSI colours, disabled automatically when output is not a terminal or NO_COLOR is set."""

    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    @classmethod
    def paint(cls, text, *codes):
        return f"\033[{';'.join(codes)}m{text}\033[0m" if cls.enabled else text


def bold(text):
    return Style.paint(text, "1")


def green(text):
    return Style.paint(text, "1", "32")


def red(text):
    return Style.paint(text, "1", "31")


def yellow(text):
    return Style.paint(text, "33")


def cyan(text):
    return Style.paint(text, "1", "36")


def status(message):
    """Rewrite the current terminal line (used for live camera and recording feedback)."""
    if Style.enabled:
        sys.stdout.write("\r\033[K" + message)
    else:
        sys.stdout.write(message + "\n")
    sys.stdout.flush()


def clear_line():
    if Style.enabled:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def end_status():
    if Style.enabled:
        sys.stdout.write("\n")
        sys.stdout.flush()


def bar(fraction, width=24):
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "█" * filled + "░" * (width - filled)


def step(number, total, title):
    print()
    print(cyan(f"[{number}/{total}] {title}"))


def display_width(text):
    """Terminal columns used by text (emoji and other wide characters take two)."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 0 if unicodedata.combining(ch) else 1
               for ch in text)


def pad(text, width):
    return text + " " * max(0, width - display_width(text))


def card(title, rows, colour):
    """Print a framed summary box."""
    width = max([display_width(title)] + [display_width(f"{k}: {v}") for k, v in rows]) + 4
    print(colour("╔" + "═" * width + "╗"))
    print(colour("║ ") + bold(pad(title, width - 2)) + colour(" ║"))
    print(colour("╟" + "─" * width + "╢"))
    for key, value in rows:
        print(colour("║ ") + pad(f"{key}: {value}", width - 2) + colour(" ║"))
    print(colour("╚" + "═" * width + "╝"))


class Spinner:
    """Animated 'working...' line while a slow step (model loading, embeddings) runs."""

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        index = 0
        while not self._stop.wait(0.1):
            status(f"{cyan(self.frames[index % len(self.frames)])} {self.message}")
            index += 1

    def __enter__(self):
        if Style.enabled:
            self._thread.start()
        else:
            print(self.message)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
            clear_line()


def show_face_preview(image_bgr, box=None, width=34):
    """Draw the captured face in the terminal with coloured half-block characters."""
    import cv2

    if not Style.enabled:
        return
    if box is not None:
        x, y, w, h = (int(v) for v in box[:4])
        margin = int(0.25 * max(w, h))
        top, left = max(0, y - margin), max(0, x - margin)
        image_bgr = image_bgr[top:y + h + margin, left:x + w + margin]
    height = max(2, int(width * image_bgr.shape[0] / image_bgr.shape[1]))
    height += height % 2
    small = cv2.cvtColor(cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    for row in range(0, height, 2):
        line = []
        for column in range(width):
            (r1, g1, b1), (r2, g2, b2) = small[row, column], small[row + 1, column]
            line.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀")
        print("      " + "".join(line) + "\033[0m")


# --------------------------------------------------------------------------- menus
class Quit(Exception):
    """The user chose to leave the application."""


def choose(question, options):
    """Ask until one of the option keys is typed. options: {key: label}. EOF counts as quit."""
    menu = "   ".join(label if key == "" else f"[{bold(key.upper())}] {label}" for key, label in options.items())
    while True:
        print(f"  {question}")
        try:
            answer = input(f"  {menu}\n  > ").strip().lower()
        except EOFError:
            answer = "q"
        if answer == "" and "" in options:
            return ""
        if answer[:1] in options:
            return answer[:1]
        print(yellow("  Please type one of the letters shown."))


def retry_or_quit(stage, tries_used, max_attempts):
    """After a failed try: return True to retry, raise Quit to leave, or end when out of tries."""
    left = max_attempts - tries_used
    if left <= 0:
        print(red(f"  ✖ Maximum of {max_attempts} {stage} attempts reached."))
        raise Quit(f"too many failed {stage} attempts")
    choice = choose(f"{left} {stage} attempt(s) left. What would you like to do?",
                    {"r": f"Retry {stage} check", "q": "Quit"})
    if choice == "q":
        raise Quit("cancelled by the user")
    return True


# --------------------------------------------------------------------------- sign in
def find_account(registry_frame, text):
    """Return the user_id for a username, display name or email (case-insensitive), or None."""
    text = text.strip().lower()
    if not text:
        return None
    for column in ("user_id", "display_name", "username", "email"):
        if column in registry_frame:
            matches = registry_frame[registry_frame[column].fillna("").str.strip().str.lower() == text]
            if len(matches) == 1:
                return matches.iloc[0]["user_id"]
    return None


def sign_in(registry_frame, attempts=3):
    has_email = "email" in registry_frame
    label = "Username or email" if has_email else "Username"
    for remaining in range(attempts - 1, -1, -1):
        try:
            text = input(f"  {label} (or Q to quit): ")
        except EOFError:
            text = "q"
        if text.strip().lower() in ("q", "quit", "exit"):
            raise Quit("cancelled by the user")
        user_id = find_account(registry_frame, text)
        if user_id:
            return user_id
        print(red("  No account found with that name.") +
              (f" {remaining} attempt(s) left." if remaining else ""))
    return None


# --------------------------------------------------------------------------- camera
class CaptureError(RuntimeError):
    """The camera or microphone could not be used, or capture was cancelled."""


def analyze_frame(detector, frame, min_face=90):
    """Detect faces in one BGR frame. Returns (ready, message, box).

    Uses the same YuNet detector as the face model, so a photo taken while ready is
    True also passes the model's own "exactly one face" check.
    """
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    count = 0 if faces is None else len(faces)
    if count == 0:
        return False, "Looking for your face... face the camera", None
    if count > 1:
        return False, f"I can see {count} faces. Only one person, please", None
    if min(faces[0, 2:4]) < min_face:
        return False, "Face found. Come a little closer", faces[0]
    return True, "Ready! Click the picture, press Space or the button", faces[0]


def open_camera(camera):
    import cv2

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise CaptureError(f"Could not open camera {camera}. Is it connected and not used by another app?")
    for _ in range(5):  # let the camera adjust exposure
        capture.read()
    return capture


def capture_face(detector_owner, destination, camera=0, timeout=25.0, steady_frames=10, min_face=90):
    """Terminal mode: save a photo automatically once one large face stays steady."""
    import cv2

    capture = open_camera(camera)
    try:
        steady, started = 0, time.time()
        while time.time() - started < timeout:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CaptureError("The camera stopped sending images")
            ready, message, box = analyze_frame(detector_owner.detector, frame, min_face)
            left = int(timeout - (time.time() - started))
            if not ready:
                steady = 0
                status(yellow(f"  👀 {message}  ({left}s)"))
                continue
            steady += 1
            status(green(f"  🙂 Hold still {bar(steady / steady_frames, 16)}"))
            if steady >= steady_frames:
                end_status()
                cv2.imwrite(str(destination), frame)
                return frame, box
        end_status()
        raise CaptureError("No clear face was seen in time")
    finally:
        capture.release()


def window_available():
    """True when a desktop display and Tkinter can show a camera window."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import tkinter  # noqa: F401
        from PIL import ImageTk  # noqa: F401
    except ImportError:
        return False
    return True


def capture_face_window(detector_owner, destination, camera=0, min_face=90, title="Face check"):
    """Window mode: a small live camera window; click it, press Space or the button to take
    the photo once one face is framed. Escape or closing the window cancels."""
    import cv2
    import tkinter as tk
    from PIL import Image, ImageTk

    capture = open_camera(camera)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        capture.release()
        raise CaptureError(f"Could not open a window ({error})")
    root.title(title)
    root.resizable(False, False)
    root.attributes("-topmost", True)
    state = {"frame": None, "box": None, "ready": False, "result": None, "error": None, "photo": None,
             "job": None}

    picture = tk.Label(root, cursor="hand2", bd=0)
    picture.pack(padx=8, pady=(8, 4))
    message = tk.Label(root, text="Starting camera...", font=("Helvetica", 12))
    message.pack(pady=2)
    buttons = tk.Frame(root)
    buttons.pack(pady=(4, 10))
    shoot = tk.Button(buttons, text="📸  Take photo", font=("Helvetica", 12, "bold"), width=14, state="disabled")
    shoot.pack(side="left", padx=6)
    cancel = tk.Button(buttons, text="Cancel", font=("Helvetica", 11), width=8)
    cancel.pack(side="left", padx=6)

    def close():
        if state["job"] is not None:
            root.after_cancel(state["job"])
            state["job"] = None
        root.destroy()

    def take(_event=None):
        if state["ready"] and state["frame"] is not None:
            cv2.imwrite(str(destination), state["frame"])
            state["result"] = (state["frame"].copy(), state["box"])
            close()
        else:
            message.config(text="Not yet: " + message.cget("text"), fg="#c0392b")

    def stop(_event=None):
        close()

    def refresh():
        ok, frame = capture.read()
        if not ok or frame is None:
            state["error"] = "The camera stopped sending images"
            state["job"] = None
            close()
            return
        ready, text, box = analyze_frame(detector_owner.detector, frame, min_face)
        state.update(frame=frame, box=box, ready=ready)
        shown = frame.copy()
        if box is not None:
            x, y, w, h = (int(v) for v in box[:4])
            cv2.rectangle(shown, (x, y), (x + w, y + h), (80, 200, 80) if ready else (0, 165, 255), 3)
        shown = cv2.flip(shown, 1)  # mirror view, like a selfie camera; the saved photo is not mirrored
        scale = 480 / shown.shape[1]
        shown = cv2.resize(shown, (480, int(shown.shape[0] * scale)))
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        picture.config(image=state["photo"])
        message.config(text=text, fg="#1e8449" if ready else "#b9770e")
        shoot.config(state="normal" if ready else "disabled")
        state["job"] = root.after(40, refresh)

    shoot.config(command=take)
    cancel.config(command=stop)
    picture.bind("<Button-1>", take)
    root.bind("<space>", take)
    root.bind("<Return>", take)
    root.bind("<Escape>", stop)
    root.protocol("WM_DELETE_WINDOW", stop)
    state["job"] = root.after(10, refresh)
    try:
        root.mainloop()
    finally:
        capture.release()
    if state["error"]:
        raise CaptureError(state["error"])
    if state["result"] is None:
        raise CaptureError("The camera window was closed without taking a photo")
    return state["result"]


# --------------------------------------------------------------------------- microphone
def record_voice(destination, seconds=4.0, rate=16000):
    """Record mono speech from the default microphone, showing a live progress bar."""
    if shutil.which("arecord"):
        command = ["arecord", "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1",
                   "-d", str(int(round(seconds))), str(destination)]
    elif shutil.which("ffmpeg"):
        command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "pulse", "-i", "default",
                   "-ac", "1", "-ar", str(rate), "-t", str(seconds), str(destination)]
    else:
        raise CaptureError("No recording tool found (install alsa-utils for 'arecord')")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    started = time.time()
    while process.poll() is None:
        elapsed = time.time() - started
        status(red("  ● REC ") + bar(elapsed / seconds) + f" {min(elapsed, seconds):.1f}s / {seconds:.0f}s")
        time.sleep(0.1)
    end_status()
    if process.returncode != 0 or not Path(destination).exists():
        detail = process.stderr.read().decode(errors="replace").strip()
        raise CaptureError(f"Recording failed: {detail or 'no audio captured'}")


def loudness(path):
    """Peak level (0..1) of a 16-bit WAV, used to catch silent recordings before the model."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return float(np.abs(samples).max()) / 32768 if samples.size else 0.0


def capture_voice(destination, name, seconds):
    print(f"  {name}, when the countdown ends, say clearly:  {bold('“' + PASSPHRASES[0] + '”')}")
    for number in (3, 2, 1):
        status(f"  Recording in {bold(str(number))}...")
        time.sleep(0.8)
    end_status()
    record_voice(destination, seconds)
    if loudness(destination) < 0.02:
        raise CaptureError("I could hardly hear you. Speak closer to the microphone")
    print(green("  ✔ Voice recorded"))


# --------------------------------------------------------------------------- checks
def check_face(ctx, user_id, path):
    """Same face check as attempt(), plus: the face must belong to the signed-in account."""
    try:
        features = ctx["extract_face"](path)
    except ValueError as error:  # no face, several faces, too small, unreadable
        raise AccessDenied(f"Face image could not be used ({error})", "face")
    identity, confidence = verify_face(ctx["face_model"], features, ctx["registry"],
                                            ctx.get("face_gate", 0.5))
    if identity != user_id:
        raise AccessDenied("The face does not match the account you signed in with", "face")
    return confidence


def check_voice(ctx, user_id, path):
    """Same voice check as attempt(): the voice must belong to the same user."""
    try:
        features = ctx["extract_voice"](path)
    except ValueError as error:  # undecodable audio or no speech after silence trimming
        raise AccessDenied(f"Voice sample could not be used ({error})", "voice")
    _, confidence = verify_voice(ctx["voice_model"], features, ctx["registry"],
                                      ctx["metadata"]["voice"]["threshold"], user_id)
    return confidence


def recommend(ctx, user_id):
    """Same product step as attempt(); only reached after both checks passed."""
    rows = ctx["customers"][ctx["customers"].customer_id == ctx["registry"][user_id]]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one inference row for customer {ctx['registry'][user_id]}")
    columns = ctx["metadata"]["product"]["feature_columns"]
    model = ctx["product_model"]
    prediction = str(model.predict(rows[columns])[0])
    confidence = float(model.predict_proba(rows[columns]).max()) if hasattr(model, "predict_proba") else None
    return prediction, confidence


def log(args, face_path, voice_path, outcome, stage, identity, reason):
    log_attempt(args.log, face_path or "", voice_path,
                     {"outcome": outcome, "stage": stage, "identity": identity, "reason": reason})


def ask_file(kind, current):
    """File mode retry: let the user pick another file (Enter keeps the current one)."""
    try:
        text = input(f"  Path to another {kind} (Enter to reuse {Path(current).name}): ").strip().strip('"\'')
    except EOFError:
        text = ""
    return Path(text) if text else Path(current)


# --------------------------------------------------------------------------- scripted mode
def _ask(prompt):
    return input(prompt).strip().strip('"').strip("'")


def run_scripted(args):
    """One non-interactive attempt from files: face, then voice, then product."""
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


# --------------------------------------------------------------------------- interactive app
def run(args):
    import pandas as pd

    total = 5
    print()
    lines = ["🔐  SECURE PRODUCT RECOMMENDATION", "    face  ·  voice  ·  personalised advice"]
    inner = max(display_width(line) for line in lines) + 4
    print(bold("  ╭" + "─" * inner + "╮"))
    for line in lines:
        print(bold("  │  " + pad(line, inner - 2) + "│"))
    print(bold("  ╰" + "─" * inner + "╯"))

    if args.weights:
        os.environ["FACE_WEIGHTS_DIR"] = str(args.weights)
    with Spinner("Loading face, voice and product models ..."):
        ctx = load_context(args)
        from feature_extraction import _encoder, _load_voice_encoder
        encoder = _encoder(str(args.weights or ROOT / "models" / "pretrained"))
        if not args.audio:
            _load_voice_encoder()  # warm up so the voice check feels instant later
    print(green("  ✔ Models ready"))
    registry_frame = pd.read_csv(args.registry, dtype=str)
    use_window = not args.photo and not args.no_window and window_available()

    workdir = Path(args.keep_captures) if args.keep_captures else Path(tempfile.mkdtemp(prefix="auth_"))
    workdir.mkdir(parents=True, exist_ok=True)
    face_path, voice_path, user_id = None, None, ""
    try:
        # 1. SIGN IN
        step(1, total, "Sign in")
        user_id = sign_in(registry_frame, args.max_attempts)
        if user_id is None:
            print(red(f"  ✖ Maximum of {args.max_attempts} sign-in attempts reached."))
            raise Quit("unknown account")
        name = ctx["names"].get(user_id, user_id)
        print(f"  Welcome, {bold(name)}. Let's confirm it's really you.")

        # 2. FACE (retry or quit after each failure)
        step(2, total, "Face check")
        face_path = Path(args.photo) if args.photo else workdir / "face_capture.jpg"
        tries = 0
        while True:
            tries += 1
            print(f"  Attempt {tries} of {args.max_attempts}")
            try:
                if args.photo:
                    print(f"  Using photo {face_path}")
                else:
                    if use_window:
                        print("  A camera window is opening: click the picture, press Space or "
                              f"{bold('📸 Take photo')} when your face is framed.")
                        frame, box = capture_face_window(encoder, face_path, camera=args.camera)
                    else:
                        if choose("Ready for the camera?", {"": "Enter: start", "q": "Quit"}) == "q":
                            raise Quit("cancelled by the user")
                        frame, box = capture_face(encoder, face_path, camera=args.camera)
                    print(green("  📸 Photo taken"))
                    show_face_preview(frame, box)
                with Spinner("Comparing your face with the enrolled faces ..."):
                    face_confidence = check_face(ctx, user_id, face_path)
                print(green(f"  ✔ Face verified: {name} ({face_confidence:.0%})"))
                break
            except (AccessDenied, CaptureError) as error:
                print(red(f"  ✖ {error}"))
                log(args, face_path, None, "denied", "face", user_id, str(error))
                retry_or_quit("face", tries, args.max_attempts)
                if args.photo:
                    face_path = ask_file("photo", face_path)

        # 3. REQUEST
        step(3, total, "Request a recommendation")
        if choose(f"{name}, request your personalised product recommendation?",
                  {"y": "Yes, continue to voice check", "q": "Quit"}) == "q":
            raise Quit("cancelled by the user")

        # 4. VOICE (retry or quit after each failure)
        step(4, total, "Voice check")
        voice_path = Path(args.audio) if args.audio else workdir / "voice_capture.wav"
        tries = 0
        while True:
            tries += 1
            print(f"  Attempt {tries} of {args.max_attempts}")
            try:
                if args.audio:
                    print(f"  Using recording {voice_path}")
                else:
                    if choose("Ready to speak?", {"": "Enter: start recording", "q": "Quit"}) == "q":
                        raise Quit("cancelled by the user")
                    capture_voice(voice_path, name, args.seconds)
                with Spinner("Comparing your voice with the enrolled voice ..."):
                    voice_confidence = check_voice(ctx, user_id, voice_path)
                print(green(f"  ✔ Voice verified: {name} ({voice_confidence:.0%})"))
                break
            except (AccessDenied, CaptureError) as error:
                print(red(f"  ✖ {error}"))
                stage = getattr(error, "stage", "voice")
                log(args, face_path, voice_path, "denied", stage, user_id, str(error))
                retry_or_quit("voice", tries, args.max_attempts)
                if args.audio:
                    voice_path = ask_file("recording", voice_path)

        # 5. PRODUCT
        step(5, total, "Your recommendation")
        with Spinner("Preparing your recommendation ..."):
            prediction, product_confidence = recommend(ctx, user_id)
        log(args, face_path, voice_path, "granted", "done", user_id, "")
        card("✅ ACCESS GRANTED", [
            ("User", name),
            ("Face match", f"{face_confidence:.0%}"),
            ("Voice match", f"{voice_confidence:.0%}"),
            ("Recommended product", f"🛒 {prediction}"),
            ("Model confidence", f"{product_confidence:.0%}" if product_confidence is not None else "n/a"),
        ], green)
        return 0
    except Quit as reason:
        print()
        card("⛔ SESSION ENDED", [("Reason", str(reason).capitalize()), ("Prediction", "not made")], red)
        return 1
    finally:
        if not args.keep_captures:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    interactive = parser.add_argument_group("interactive app (default)")
    interactive.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    interactive.add_argument("--no-window", action="store_true",
                             help="no camera window: take the photo automatically in the terminal")
    interactive.add_argument("--seconds", type=float, default=4.0, help="voice recording length in seconds")
    interactive.add_argument("--max-attempts", type=int, default=3, help="tries allowed per step (default 3)")
    interactive.add_argument("--photo", type=Path, help="use this face photo instead of the webcam")
    interactive.add_argument("--audio", type=Path, help="use this voice recording instead of the microphone")
    interactive.add_argument("--keep-captures", type=Path,
                             help="folder to keep the captured photo and recording (deleted by default)")
    interactive.add_argument("--weights", type=Path,
                             help="folder with the face ONNX files (default models/pretrained)")
    scripted = parser.add_argument_group("scripted single attempt")
    scripted.add_argument("--face", type=Path, help="face image file (runs one scripted attempt)")
    scripted.add_argument("--voice", type=Path, help="voice file (asked for after the face passes if omitted)")
    scripted.add_argument("--yes", action="store_true", help="skip the 'press Enter' request step")
    shared = parser.add_argument_group("files")
    shared.add_argument("--models", type=Path, default=ROOT / "models")
    shared.add_argument("--registry", type=Path, default=ROOT / "data/processed/user_registry.csv")
    shared.add_argument("--customers", type=Path, default=ROOT / "data/processed/customer_inference.csv")
    shared.add_argument("--log", type=Path, default=ROOT / "reports/access_log.csv")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    if args.face or args.voice:
        try:
            parser.exit(run_scripted(args))
        except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as error:
            parser.exit(2, f"Setup error: {error}\n")

    try:
        code = run(args)
    except KeyboardInterrupt:
        print(red("\n  Cancelled."))
        code = 1
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(red(f"\n  Setup problem: {error}"))
        code = 2
    print()
    sys.exit(code)


if __name__ == "__main__":
    main()
