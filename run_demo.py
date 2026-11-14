"""Run the scripted demo scenarios through the same code path as main.py.

    python run_demo.py                       # all scenarios in data/processed/demo_scenarios.csv
    python run_demo.py --scenarios my.csv

Writes reports/integration_results.csv and prints a pass/fail table. A scenario whose sample
file does not exist yet is reported as SKIPPED (never as a pass). Exit code 1 if any
scenario did not behave as expected.

Special values in the face/voice columns create a throw-away test file:
    @blank_image   a plain white picture (no face)
    @silence       one second of silent audio (no speech)
"""
import argparse
import sys
import tempfile
import wave
from pathlib import Path

import pandas as pd

import main as app

ROOT = Path(__file__).resolve().parent


def make_special(token, folder):
    """Create the file for an @token and return its path."""
    folder = Path(folder)
    if token == "@blank_image":
        from PIL import Image
        path = folder / "blank_image.png"
        Image.new("RGB", (400, 400), "white").save(path)
    elif token == "@silence":
        path = folder / "silence.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)
    else:
        raise ValueError(f"Unknown special value {token}")
    return path


def run_scenarios(ctx, scenarios, root=ROOT, say=lambda *_: None):
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        for s in scenarios.itertuples():
            face = make_special(s.face, folder) if s.face.startswith("@") else root / s.face
            voice = make_special(s.voice, folder) if s.voice.startswith("@") else root / s.voice
            row = {"scenario": s.scenario, "expected": f"{s.expect_outcome}@{s.expect_stage}"}
            missing = [str(p.relative_to(root)) for p in (face, voice)
                       if not p.exists() and root in p.parents]
            if missing:
                row.update(actual="-", status="SKIPPED", detail="missing file: " + ", ".join(missing))
            else:
                r = app.attempt(ctx, face, voice, say=say)
                ok = r["outcome"] == s.expect_outcome and r["stage"] == s.expect_stage
                # Hard safety check: a denied attempt must never carry a prediction.
                ok = ok and not (r["outcome"] == "denied" and r["prediction"])
                row.update(actual=f"{r['outcome']}@{r['stage']}", status="PASS" if ok else "FAIL",
                           detail=r["prediction"] or r["reason"], identity=r["identity"],
                           face_conf=r["confidence_face"], voice_conf=r["confidence_voice"])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", type=Path, default=ROOT / "data/processed/demo_scenarios.csv")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--registry", type=Path, default=ROOT / "data/processed/user_registry.csv")
    p.add_argument("--customers", type=Path, default=ROOT / "data/processed/customer_inference.csv")
    p.add_argument("--out", type=Path, default=ROOT / "reports/integration_results.csv")
    args = p.parse_args()
    try:
        ctx = app.load_context(args)
    except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as error:
        p.exit(2, f"Setup error: {error}\n")
    table = run_scenarios(ctx, pd.read_csv(args.scenarios, dtype=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    counts = table.status.value_counts().to_dict()
    print(f"\n{counts}  -> saved to {args.out}")
    sys.exit(1 if counts.get("FAIL") else 0)


if __name__ == "__main__":
    main()
