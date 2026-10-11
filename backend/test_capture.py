# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Post a synthetic capture to the BrainRot backend and print the response + latency.

  uv run backend/test_capture.py --url https://<...>.modal.run --scenario nature
Scenarios: nature (landscape pan + no speech), talk (speech audio + testsrc), desk (static grey UI-ish
frames + silence), audio_only (speech, no frames), video_only (nature frames, no audio).
Uses only local assets: macOS desktop pictures + `say` for speech, ffmpeg for everything else.
"""
import argparse
import glob
import json
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

SPEECH = ("Hey, did you see the game last night? I can't believe they lost in overtime. "
          "Anyway, want to grab lunch later? I was thinking that new taco place on Main Street.")


def sh(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def nature_source():
    pref = ["Big Sur Mountains", "Big Sur Coastline", "Catalina Coast", "Big Sur Aerial"]
    cands = [p for n in pref for p in glob.glob(f"/System/Library/Desktop Pictures/.thumbnails/{n}.heic")] + \
        sorted(glob.glob("/System/Library/Desktop Pictures/*.heic")) + sorted(glob.glob("/Library/Desktop Pictures/*.heic"))
    return cands[0] if cands else None


def make_frames(d: Path, kind: str, n=25, fps=7):
    out = d / "frames"
    out.mkdir()
    if kind == "nature" and nature_source():
        src = d / "nature.jpg"
        sh(["sips", "-s", "format", "jpeg", nature_source(), "--out", str(src)])
        vf = (f"scale=1600:-2,zoompan=z='1.0+0.004*on':x='iw/2-(iw/zoom/2)+on*6':y='ih/2-(ih/zoom/2)'"
              f":d={n}:s=360x640:fps={fps}")
        sh(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-frames:v", str(n), "-q:v", "4", str(out / "%03d.jpg")])
    elif kind == "desk":
        sh(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x2b2d31:s=360x640:r={fps}:d=4",
            "-vf", "drawbox=x=20:y=100:w=320:h=40:color=white@0.6:t=fill,drawbox=x=20:y=180:w=200:h=20:color=white@0.4:t=fill",
            "-frames:v", str(n), "-q:v", "4", str(out / "%03d.jpg")])
    else:
        sh(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=s=360x640:r={fps}:d=4", "-frames:v", str(n),
            "-q:v", "4", str(out / "%03d.jpg")])
    return sorted(out.glob("*.jpg"))


def make_audio(d: Path, speech: bool):
    wav = d / "audio.wav"
    if speech:
        aiff = d / "s.aiff"
        sh(["say", "-o", str(aiff), SPEECH])
        sh(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", "-t", "15", str(wav)])
    else:
        sh(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "15", str(wav)])
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--scenario", default="talk",
                    choices=["nature", "talk", "desk", "audio_only", "video_only"])
    ap.add_argument("--timestamp", default=None, help="ISO8601; default now")
    ap.add_argument("--fps", type=int, default=7)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    d = Path(tempfile.mkdtemp())
    frames, wav = [], None
    if a.scenario in ("nature", "video_only"):
        frames = make_frames(d, "nature")
    elif a.scenario == "desk":
        frames = make_frames(d, "desk")
    elif a.scenario == "talk":
        frames = make_frames(d, "testsrc")
    if a.scenario in ("talk", "audio_only"):
        wav = make_audio(d, speech=True)
    elif a.scenario in ("desk", "nature"):
        wav = make_audio(d, speech=False)

    files = [("frames", (p.name, p.read_bytes(), "image/jpeg")) for p in frames]
    if wav:
        files.append(("audio", ("audio.wav", wav.read_bytes(), "audio/wav")))
    data = {"fps": str(a.fps), "capture_id": str(uuid.uuid4()),
            "timestamp": a.timestamp or datetime.now(timezone.utc).isoformat(), "glasses_battery": "77"}
    print(f"POST {url}/capture scenario={a.scenario} frames={len(frames)} audio={bool(wav)}")
    t0 = time.time()
    r = requests.post(f"{url}/capture", data=data, files=files or None, timeout=900)
    dt = time.time() - t0
    print("status", r.status_code, f"latency {dt:.1f}s")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:2000])


if __name__ == "__main__":
    main()
