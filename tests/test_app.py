"""Interactive app helpers: sign-in lookup, automatic camera capture and silence detection."""
import os
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "pretrained"


def registry(with_email=False):
    frame = pd.DataFrame({"user_id": ["maurice", "larissa"], "display_name": ["Maurice", "Larissa"],
                          "customer_id": ["121", "120"]})
    if with_email:
        frame["email"] = ["maurice@example.com", ""]
    return frame


def test_sign_in_accepts_username_display_name_and_email():
    assert app.find_account(registry(), "maurice") == "maurice"
    assert app.find_account(registry(), "  LARISSA ") == "larissa"
    assert app.find_account(registry(), "bob") is None
    assert app.find_account(registry(), "") is None
    assert app.find_account(registry(with_email=True), "Maurice@Example.com") == "maurice"


def test_sign_in_gives_up_after_three_unknown_names():
    with patch("builtins.input", side_effect=["bob", "alice", "eve"]):
        assert app.sign_in(registry()) is None
    with patch("builtins.input", side_effect=["bob", "larissa"]):
        assert app.sign_in(registry()) == "larissa"


def write_wav(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_loudness_flags_silent_recordings(tmp_path):
    write_wav(tmp_path / "silent.wav", np.zeros(16000))
    write_wav(tmp_path / "speech.wav", (8000 * np.sin(np.linspace(0, 400, 16000))).astype(int))
    assert app.loudness(tmp_path / "silent.wav") < 0.02
    assert app.loudness(tmp_path / "speech.wav") > 0.2


class FakeCamera:
    """Stands in for cv2.VideoCapture and replays fixed frames."""

    def __init__(self, frames):
        self.frames = list(frames)

    def isOpened(self):
        return True

    def read(self):
        return True, self.frames[0] if len(self.frames) == 1 else self.frames.pop(0)

    def release(self):
        pass


@pytest.mark.skipif(not (WEIGHTS / "face_detection_yunet_2023mar.onnx").is_file(), reason="face weights missing")
def test_camera_capture_waits_for_a_steady_face_and_saves_it(tmp_path):
    import cv2
    from feature_extraction import _encoder

    encoder = _encoder(str(WEIGHTS))
    face = cv2.imread(str(ROOT / "data/raw/images/maurice/Maurice_Neutral.jpeg"))
    face = cv2.resize(face, (480, 640))
    blank = np.full_like(face, 200)
    frames = [blank] * 8 + [face]  # 5 warm-up frames, then no face, then a face that stays
    with patch("cv2.VideoCapture", return_value=FakeCamera(frames)):
        frame, box = app.capture_face(encoder, tmp_path / "shot.jpg", steady_frames=3, min_face=40, timeout=10)
    assert (tmp_path / "shot.jpg").is_file()
    assert frame.shape == face.shape and len(box) >= 4


@pytest.mark.skipif(not (WEIGHTS / "face_detection_yunet_2023mar.onnx").is_file(), reason="face weights missing")
def test_camera_capture_times_out_without_a_face(tmp_path):
    from feature_extraction import _encoder

    encoder = _encoder(str(WEIGHTS))
    with patch("cv2.VideoCapture", return_value=FakeCamera([np.full((480, 640, 3), 200, np.uint8)])):
        with pytest.raises(app.CaptureError, match="No clear face"):
            app.capture_face(encoder, tmp_path / "shot.jpg", timeout=0.5)
    assert not (tmp_path / "shot.jpg").exists()


def test_sign_in_quit_option():
    with patch("builtins.input", side_effect=["q"]):
        with pytest.raises(app.Quit):
            app.sign_in(registry())


def test_retry_menu_retries_quits_and_stops_at_the_limit():
    with patch("builtins.input", side_effect=["r"]):
        assert app.retry_or_quit("face", 1, 3) is True
    with patch("builtins.input", side_effect=["x", "q"]):  # invalid key, then quit
        with pytest.raises(app.Quit, match="cancelled"):
            app.retry_or_quit("voice", 1, 3)
    with pytest.raises(app.Quit, match="too many failed face attempts"):
        app.retry_or_quit("face", 3, 3)  # no prompt once the limit is reached


def test_menu_treats_end_of_input_as_quit():
    with patch("builtins.input", side_effect=EOFError):
        assert app.choose("Continue?", {"y": "Yes", "q": "Quit"}) == "q"


@pytest.mark.skipif(not (WEIGHTS / "face_detection_yunet_2023mar.onnx").is_file(), reason="face weights missing")
def test_frame_analysis_reports_missing_small_and_ready_faces():
    import cv2
    from feature_extraction import _encoder

    detector = _encoder(str(WEIGHTS)).detector
    face = cv2.resize(cv2.imread(str(ROOT / "data/raw/images/maurice/Maurice_Neutral.jpeg")), (480, 640))
    ready, message, box = app.analyze_frame(detector, np.full_like(face, 200))
    assert not ready and "Looking" in message and box is None
    ready, message, box = app.analyze_frame(detector, face, min_face=40)
    assert ready and box is not None
    ready, message, _ = app.analyze_frame(detector, face, min_face=500)
    assert not ready and "closer" in message


@pytest.mark.skipif(os.environ.get("AUTH_GUI_TESTS") != "1" or not app.window_available()
                    or not (WEIGHTS / "face_detection_yunet_2023mar.onnx").is_file(),
                    reason="set AUTH_GUI_TESTS=1 on a desktop to test the camera window")
def test_camera_window_ignores_early_clicks_then_takes_photo(tmp_path):
    import tkinter as tk
    import cv2
    from feature_extraction import _encoder

    face = cv2.resize(cv2.imread(str(ROOT / "data/raw/images/maurice/Maurice_Neutral.jpeg")), (480, 640))
    frames = [np.full_like(face, 200)] * 14 + [face]

    class AutoTk(tk.Tk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after(300, lambda: self.event_generate("<space>"))   # no face yet: ignored
            self.after(1500, lambda: self.event_generate("<space>"))  # face visible: photo taken

    with patch("cv2.VideoCapture", return_value=FakeCamera(frames)), patch("tkinter.Tk", AutoTk):
        frame, box = app.capture_face_window(_encoder(str(WEIGHTS)), tmp_path / "shot.jpg")
    assert (tmp_path / "shot.jpg").is_file() and box is not None
