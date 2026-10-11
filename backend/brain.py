"""BrainRot core: TRIBE v2 inference engine, rot state store, Gemini roasts, FastAPI app."""
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("brainrot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _n in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

REGION_MAP = json.loads(Path("/root/region_map.json").read_text()) if Path("/root/region_map.json").exists() else json.loads((Path(__file__).parent / "region_map.json").read_text())
REGIONS = REGION_MAP["regions"]
VERTEX_REGION = np.asarray(REGION_MAP["region"])
VISUAL_REGIONS = {"nature_places", "faces_social", "movement", "reading"}

# ---------------- tunables (env-configurable) ----------------
def _f(name, default):
    return float(os.environ.get(name, default))

HALF_LIFE_MIN = _f("HALF_LIFE_MIN", 5)
GAIN = _f("GAIN", 0.15)
THETA = _f("THETA", 1.0)
H_FLOOR = _f("H_FLOOR", 0.2)  # health decays toward this (rot asymptote 1-H_FLOOR) instead of 0
MAX_HEAL = _f("MAX_HEAL", 0.4)  # per capture, per region
MIN_SD = _f("MIN_SD", 0.3)  # floor on session sd so small indoor wiggles don't look huge
H0 = _f("H0", 0.7)
DECAY_MULT = {"nature_places": 1.3, "faces_social": 1.0, "movement": 0.9, "reading": 1.1, "sound_speech": 0.8, "language": 1.0}
# prior mean / sd of the cross-cortex-normalised region score (calibrated on test clips)
# Typical cross-cortex-normalised scores for an everyday indoor clip with phone audio
# (from calibration runs: speech lifts auditory/language, no-audio clips push them negative).
# Calibrated on the first real indoor captures from the glasses (video + phone audio).
PRIOR_MU = {"nature_places": 1.0, "faces_social": -0.15, "movement": 0.5, "reading": 0.85,
            "sound_speech": 0.2, "language": -0.75}
PRIOR_MU.update(json.loads(os.environ.get("PRIOR_MU", "{}")))
PRIOR_SD = _f("PRIOR_SD", 0.35)
PRIOR_N = _f("PRIOR_N", 6)
ROT_THRESH = 0.6
REARM_THRESH = 0.55
REGION_COOLDOWN_MIN = _f("REGION_COOLDOWN_MIN", 5)
GLOBAL_GAP_MIN = _f("GLOBAL_GAP_MIN", 2)
HEAL_ROAST_DROP = _f("HEAL_ROAST_DROP", 0.15)
VIDEO_S = _f("VIDEO_S", 4)  # frames are looped to at least this many seconds
MAX_CLIP_S = _f("MAX_CLIP_S", 10)
MAX_AUDIO_S = _f("MAX_AUDIO_S", 20)
SCORE_MODE = os.environ.get("SCORE_MODE", "norm")
AUTOCAST = os.environ.get("AUTOCAST", "0") == "1"
LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "America/Los_Angeles"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"]
GEMINI_BUDGET_S = _f("GEMINI_BUDGET_S", 15)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
CACHE_DIR = "/tmp/tribe_cache"
# Hybrid scoring: z = W_TRIBE * z_tribe + (1 - W_TRIBE) * z_sem,  z_sem = (s - SEM_CENTER) * SEM_SCALE
W_TRIBE = _f("W_TRIBE", 0.35)
SEM_CENTER = _f("SEM_CENTER", 0.35)
SEM_SCALE = _f("SEM_SCALE", 6.0)
# 3.6-flash measured ~2 s vs 3.8-flash 4-10 s (+ read timeouts); also keeps perceive and roasts in
# separate free-tier per-model RPM buckets (limit 5 RPM/model).
PERCEIVE_MODEL = os.environ.get("PERCEIVE_MODEL", "gemini-3.5-flash-lite")
PERCEIVE_BUDGET_S = _f("PERCEIVE_BUDGET_S", 12)
TUNABLE = ["HALF_LIFE_MIN", "GAIN", "THETA", "H_FLOOR", "MAX_HEAL", "MIN_SD", "PRIOR_SD", "PRIOR_N", "PRIOR_MU",
           "DECAY_MULT", "REGION_COOLDOWN_MIN", "GLOBAL_GAP_MIN", "HEAL_ROAST_DROP", "GEMINI_MODEL", "VIDEO_S",
           "GEMINI_BUDGET_S", "W_TRIBE", "SEM_CENTER", "SEM_SCALE", "PERCEIVE_MODEL", "PERCEIVE_BUDGET_S"]


def set_params(p: dict) -> dict:
    """Live-tune module parameters (persisted in state so they survive restarts)."""
    g = globals()
    for k, v in (p or {}).items():
        if k not in TUNABLE:
            continue
        if isinstance(g[k], dict):
            g[k].update(v)
        else:
            g[k] = type(g[k])(v)
    return {k: g[k] for k in TUNABLE}
HALLUCINATIONS = {"thank you", "thanks for watching", "you", "thank you for watching", "bye", "okay",
                  "so", "uh", "um", "subtitles by the amara.org community"}


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r


def media_duration(path) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def max_volume_db(path) -> float:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=60)
    for line in r.stderr.splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].split("dB")[0])
            except ValueError:
                pass
    return -91.0


# ---------------- TRIBE engine ----------------
class Engine:
    def __init__(self):
        self.model = None
        self.whisper = None
        self.lock = threading.Lock()
        self._last_transcript = None

    def load(self):
        t0 = time.time()
        import torch  # noqa: F401  (loads cudnn/cublas for ctranslate2)
        from faster_whisper import WhisperModel
        from tribev2.demo_utils import TribeModel
        import tribev2.eventstransforms as et

        self.whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16",
                                    download_root=os.path.join(os.environ.get("HF_HOME", "/tmp"), "whisper"))
        engine = self

        def _transcribe(wav_filename, language):
            import pandas as pd
            segs, _ = engine.whisper.transcribe(str(wav_filename), language="en", word_timestamps=True,
                                                vad_filter=True, beam_size=1)
            words, texts = [], []
            for i, seg in enumerate(segs):
                if seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.0:
                    continue
                if seg.text.strip().lower().strip(".!? ") in HALLUCINATIONS:
                    continue
                sentence = seg.text.replace('"', "").strip()
                texts.append(sentence)
                for w in seg.words or []:
                    words.append({"text": w.word.replace('"', "").strip(), "start": w.start,
                                  "duration": max(0.01, w.end - w.start), "sequence_id": i, "sentence": sentence})
            engine._last_transcript = " ".join(texts).strip() or None
            return pd.DataFrame(words)

        et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_transcribe)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # neuralset re-instantiates the V-JEPA2 wrapper on every call -> cache it
        import neuralset.extractors.video as nv
        orig = nv._HFVideoModel
        if not getattr(orig, "_brainrot_cached", False):
            class Cached(orig):
                _brainrot_cached = True
                _cache = {}

                def __new__(cls, *a, **kw):
                    key = (tuple(map(str, a)), tuple(sorted((k, str(v)) for k, v in kw.items())))
                    if key not in cls._cache:
                        cls._cache[key] = object.__new__(cls)
                    return cls._cache[key]

                def __init__(self, *a, **kw):
                    if getattr(self, "_brainrot_inited", False):
                        return
                    super().__init__(*a, **kw)
                    self._brainrot_inited = True

            nv._HFVideoModel = Cached
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
        log.info("engine loaded in %.1fs", time.time() - t0)

    def warmup(self):
        """Run one synthetic clip with speech so every feature extractor loads/downloads."""
        t0 = time.time()
        d = Path(tempfile.mkdtemp(prefix="warm_"))
        speech = d / "speech.mp3"
        try:
            from gtts import gTTS
            gTTS("Hello there. This is a warm up sentence so the language model loads its weights.").save(str(speech))
            a_in = ["-i", str(speech)]
        except Exception as e:
            log.warning("gTTS failed (%s), using sine", e)
            a_in = ["-f", "lavfi", "-i", "sine=frequency=440:duration=8"]
        clip, wav = d / "warm.mp4", d / "warm.wav"
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=360x640:rate=16:duration=6", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(clip)])
        run(["ffmpeg", "-y", *a_in, "-ac", "1", "-ar", "16000", "-t", "10", str(wav)])
        out = self.predict(video_path=clip, audio_path=wav)
        log.info("warmup done in %.1fs: T=%s transcript=%r", time.time() - t0, out["n_trs"], out["transcript"])
        shutil.rmtree(d, ignore_errors=True)

    def predict(self, video_path=None, audio_path=None, video_s=None, on_events=None) -> dict:
        """Video (no audio track) and/or audio wav are fed as separate events on one timeline."""
        import pandas as pd
        import torch
        from tribev2.demo_utils import get_audio_and_text_events

        rows = []
        if video_path:
            rows.append({"type": "Video", "filepath": str(video_path), "start": 0,
                         "timeline": "default", "subject": "default"})
        if audio_path:
            rows.append({"type": "Audio", "filepath": str(audio_path), "start": 0,
                         "timeline": "default", "subject": "default"})
        base = pd.DataFrame(rows)

        def _predict(ev):
            if AUTOCAST:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.model.predict(events=ev, verbose=False)
            return self.model.predict(events=ev, verbose=False)

        with self.lock:
            self._last_transcript = None
            t0 = time.time()
            try:
                events = get_audio_and_text_events(base.copy())
            except Exception as e:
                log.warning("text pipeline failed (%r); retrying audio/video only", e)
                events = get_audio_and_text_events(base.copy(), audio_only=True)
            t1 = time.time()
            if on_events is not None:  # transcript is known now -> start Gemini perceive in parallel
                try:
                    on_events(self._last_transcript)
                except Exception as e:
                    log.warning("on_events failed: %r", e)
            try:
                preds, segments = _predict(events)
            except Exception as e:
                if not (events.type == "Word").any():
                    raise
                log.exception("predict with text failed (%r); retrying without Word events", e)
                events = events.loc[events.type != "Word"].reset_index(drop=True)
                preds, segments = _predict(events)
            t2 = time.time()
        transcript = self._last_transcript
        if transcript is None and (events.type == "Word").any():
            transcript = " ".join(events.loc[events.type == "Word", "text"].astype(str)) or None
        preds = np.asarray(preds, dtype=np.float32)
        starts = np.array([float(getattr(sg, "start", i)) for i, sg in enumerate(segments)])
        log.info("tribe: events %.1fs predict %.1fs preds %s starts %s", t1 - t0, t2 - t1, preds.shape,
                 starts.round(1).tolist())
        return {"preds": preds, "starts": starts, "n_trs": int(preds.shape[0]), "transcript": transcript,
                "t_events": t1 - t0, "t_predict": t2 - t1}


def region_scores(preds: np.ndarray, starts=None, video_s=None) -> tuple[dict, dict]:
    """Per-region mean predicted activation. Returns (normalised, absolute).
    normalised = region mean of the cross-cortex z-scored map (relative to the rest of cortex);
    absolute = raw region mean. Visual regions use only TRs inside the video window."""
    m = np.ones(len(preds), bool)
    if starts is not None and video_s:
        m = starts < video_s
        if m.sum() < 1:
            m = np.ones(len(preds), bool)
    norm, absl = {}, {}
    for sel_name, sel in (("vis", m), ("all", np.ones(len(preds), bool))):
        v = preds[sel].mean(0)
        vz = (v - v.mean()) / (v.std() + 1e-8)
        for i, k in enumerate(REGIONS):
            if (k in VISUAL_REGIONS) == (sel_name == "vis"):
                norm[k] = float(vz[VERTEX_REGION == i].mean())
                absl[k] = float(v[VERTEX_REGION == i].mean())
    return norm, absl


# ---------------- clip assembly ----------------
def build_inputs(workdir: Path, frames: list[bytes], fps: int, audio: bytes | None, audio_name: str | None):
    """Returns (video_path | None, audio_path | None, info)."""
    info = {}
    wav = None
    if audio:
        raw = workdir / ("audio_in" + (Path(audio_name).suffix if audio_name else ".m4a"))
        raw.write_bytes(audio)
        wav = workdir / "audio.wav"
        try:
            run(["ffmpeg", "-y", "-i", str(raw), "-ac", "1", "-ar", "16000", "-t", str(MAX_AUDIO_S), str(wav)])
            info["audio_s"] = media_duration(wav)
            info["audio_max_db"] = max_volume_db(wav)
            if info["audio_s"] < 0.3 or info["audio_max_db"] < -45:
                info["audio_dropped"] = "silent"
                wav = None
        except Exception as e:
            info["audio_dropped"] = f"decode failed: {e}"[:200]
            wav = None
    if not frames:
        return None, wav, info
    for i, b in enumerate(frames):
        (workdir / f"f_{i:04d}.jpg").write_bytes(b)
    fps = max(1, min(int(fps or 7), 60))
    vid_s = len(frames) / fps
    clip = workdir / "clip.mp4"
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=16,format=yuv420p"
    common = ["-loop", "1", "-framerate", str(fps), "-i", str(workdir / "f_%04d.jpg")]
    T = min(MAX_CLIP_S, max(VIDEO_S, vid_s))
    run(["ffmpeg", "-y", *common, "-vf", vf, "-t", f"{T:.2f}", "-an", "-c:v", "libx264",
         "-preset", "veryfast", str(clip)])
    info.update(video_s=vid_s, clip_s=T, n_frames=len(frames))
    return clip, wav, info


# ---------------- roasts ----------------
ROAST_SYSTEM = """You are BrainRot, a rude, condescending, razor-witty roast comic who is also a neuroanatomist.
You watch the user's life through their smart glasses and a model of their predicted brain activity.
Rules:
- Insult ONLY the user (never other people in frame, never protected traits). Mean but clever, PG-13.
- Name the actual brain area with anatomical precision (e.g. parahippocampal place area, retrosplenial cortex,
  fusiform face area, superior temporal sulcus, area MT/V5, visual word form area, Heschl's gyrus, Broca's area / left IFG).
- For ROT roasts: mock what they're doing (use the image/transcript), then give ONE concrete fix
  ("go look at a tree", "talk to an actual human", "read a book", "go for a walk", "listen to music").
- For HEAL roasts: grudging, backhanded praise. Never fully nice.
- One sentence or two. MAX 140 characters. No hashtags, at most one emoji, no quotes around the output.
Examples:
- go outside and look at a tree, your parahippocampal gyrus is decomposing
- your fusiform face area forgot what a human looks like. call someone. anyone.
- Broca's area hasn't fired since breakfast. say a word out loud, any word
- wow, a tree. your retrosplenial cortex is crying tears of mild relief
- area MT thinks you died. stand up and move something other than your thumb
- a book?? your visual word form area is in shock. don't strain it
"""

REGION_HUMAN = {
    "nature_places": "scene/place network (parahippocampal place area, retrosplenial cortex, occipital place area)",
    "faces_social": "face & social network (fusiform face area, superior temporal sulcus, TPJ)",
    "movement": "motion/body network (area MT+/V5, extrastriate body area)",
    "reading": "visual word form area (left occipitotemporal sulcus)",
    "sound_speech": "auditory cortex (Heschl's gyrus, superior temporal gyrus, planum temporale)",
    "language": "language network (Broca's area / left IFG, left temporal cortex)",
}

CANNED = {
    "rot": {
        "nature_places": "your parahippocampal place area is composting itself. go look at a tree",
        "faces_social": "your fusiform face area forgot what humans look like. talk to one",
        "movement": "area MT thinks you're a houseplant. get up and move",
        "reading": "your visual word form area is fossilising. read literally one page",
        "sound_speech": "Heschl's gyrus is bored to death. go listen to something real",
        "language": "Broca's area has filed for unemployment. say a sentence to a human",
    },
    "heal": "fine, your {r} twitched back to life. don't let it go to your head",
}


# Free-tier Gemini keys get ~20 requests/day/model, so rotate across models and remember exhausted ones.
ROAST_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-3.5-flash-lite",
                "gemini-3.6-flash", "gemini-3.1-flash-lite-preview"]
PERCEIVE_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-3.5-flash",
                   "gemini-3.6-flash", "gemini-3.1-flash-lite-preview"]
_MODEL_BLOCKED: dict[str, float] = {}


def _gen(primary: str, pool: list[str], contents, cfg: dict, budget_s: float, tag: str):
    """generate_content over a rotating model list. Returns (response, model) or (None, None)."""
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, None
    t0 = time.time()
    models = [primary] + [m for m in pool if m != primary]
    for m in models:
        if _MODEL_BLOCKED.get(m, 0) > time.time():
            continue
        for thinking_off in (True, False):
            left = budget_s - (time.time() - t0)
            if left < 3:
                return None, None
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(max(10, min(left, 12)) * 1000)))
            c = dict(cfg)
            if thinking_off:
                c["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            try:
                r = client.models.generate_content(model=m, contents=contents, config=types.GenerateContentConfig(**c))
                log.info("%s %s ok in %.1fs", tag, m, time.time() - t0)
                return r, m
            except Exception as e:
                msg = repr(e)
                log.warning("%s %s (thinking_off=%s) failed: %s", tag, m, thinking_off, msg[:200])
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    _MODEL_BLOCKED[m] = time.time() + (6 * 3600 if "PerDay" in msg else 60)
                    break
                if "404" in msg or "NOT_FOUND" in msg:
                    _MODEL_BLOCKED[m] = time.time() + 24 * 3600
                    break
                if "503" in msg or "UNAVAILABLE" in msg or "504" in msg or "imeout" in msg or "timed out" in msg:
                    _MODEL_BLOCKED[m] = time.time() + 120
                    break
                if thinking_off and "400" in msg:
                    continue  # maybe thinking config unsupported -> retry once without it
                break
    return None, None


_ROAST_POOL = None


def gemini_roast(kind, region, rot_table, transcript, local_time, image: bytes | None, scene=None):
    """Hard wall-clock cap: if Gemini doesn't answer within GEMINI_BUDGET_S, use a canned rude line."""
    import concurrent.futures as cf
    global _ROAST_POOL
    if _ROAST_POOL is None:
        _ROAST_POOL = cf.ThreadPoolExecutor(max_workers=4)
    fut = _ROAST_POOL.submit(_gemini_roast, kind, region, rot_table, transcript, local_time, image, scene)
    try:
        return fut.result(timeout=GEMINI_BUDGET_S)
    except Exception as e:
        log.warning("gemini hard timeout/failure: %r", e)
        return _canned(kind, region), "canned:hard-timeout"


def _canned(kind, region):
    if kind == "rot":
        return CANNED["rot"].get(region, "your brain is rotting. go outside")
    return CANNED["heal"].format(r=REGION_HUMAN.get(region, region).split("(")[0].strip())


def _gemini_roast(kind, region, rot_table, transcript, local_time, image: bytes | None, scene=None):
    region_h = REGION_HUMAN.get(region, region)
    fallback = CANNED["rot"].get(region, "your brain is rotting. go outside") if kind == "rot" else \
        CANNED["heal"].format(r=region_h.split("(")[0].strip())
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return fallback, "canned:no-key"
    from google import genai
    from google.genai import types

    table = "\n".join(f"- {k}: rot={v:.2f}" for k, v in rot_table.items())
    prompt = (f"Roast type: {kind.upper()}\nTarget region: {region} = {region_h}\nLocal time: {local_time}\n"
              f"Current rot per region (0=healthy, 1=mush):\n{table}\n"
              f"Scene (from the glasses): {scene or 'unknown'}\n"
              f"What they heard/said (transcript): {transcript or '(silence)'}\n"
              "The attached image is what they are looking at right now (may be absent).\n"
              "Write the push notification now. Max 140 chars.")
    parts = []
    if image:
        parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
    parts.append(prompt)
    r, m = _gen(GEMINI_MODEL, ROAST_MODELS, parts,
                dict(system_instruction=ROAST_SYSTEM, temperature=1.0, max_output_tokens=2048),
                GEMINI_BUDGET_S, "roast")
    txt = ((r.text if r is not None else "") or "").strip().strip('"').strip()
    if txt:
        if len(txt) > 160:
            txt = txt[:157].rsplit(" ", 1)[0] + "..."
        return txt, m
    return fallback, "canned:gemini-failed"


# ---------------- perceive (Gemini semantic scene scoring) ----------------
PERCEIVE_SYSTEM = """You score what a person wearing smart-glasses is experiencing, for six brain-network
categories, each 0.0-1.0. Be calibrated and literal; judge the person's actual surroundings, not decorations.
Rubric:
- nature_places: trees, plants, sky, parks, open outdoor scenes / navigable places. Indoor desk/office/room ~0.1
  EVEN IF a window is visible; indoor with big plants ~0.2; urban street/parking lot without greenery ~0.4;
  urban street with trees ~0.6-0.8; park/forest/beach/trail ~0.9-1.0.
- faces_social: real human faces in view, especially close-up or eye contact (people in the room/street).
  Faces only on a screen/photo count at most 0.3. Nobody visible ~0.0; distant pedestrians ~0.3-0.5;
  face-to-face conversation partner ~0.8-1.0.
- movement: ego-motion (walking, driving, the camera moving through space) and moving people/vehicles.
  Static desk ~0.05; people walking by ~0.5; the wearer walking/cycling ~0.7-1.0.
- reading: reading physical text, books, paper, signs up close. A laptop/phone screen of text ~0.3 max.
- sound_speech: rich environmental sound, music, or speech, judged from the transcript and the setting.
  Silent/empty transcript in a quiet room ~0.1; busy street/cafe ~0.5; music or lots of speech ~0.8+.
- language: an actual conversation directed at or by the wearer, judged from the transcript. No transcript /
  silence = 0.0; a few stray words or background TV ~0.2; real back-and-forth conversation ~0.8-1.0.
If no images are provided, set nature_places, faces_social, movement and reading to null.
Also return `scene`: a terse <=12-word description of where they are and what they're doing."""

_PERCEIVE_POOL = None


def _perceive_pool():
    import concurrent.futures as cf
    global _PERCEIVE_POOL
    if _PERCEIVE_POOL is None:
        _PERCEIVE_POOL = cf.ThreadPoolExecutor(max_workers=8)
    return _PERCEIVE_POOL


def perceive_async(images: list[bytes], transcript):
    return _perceive_pool().submit(_perceive, images, transcript)


def perceive_result(fut):
    """Wait for a perceive future with a hard cap; None on failure (-> TRIBE-only scoring)."""
    if fut is None:
        return None
    try:
        return fut.result(timeout=PERCEIVE_BUDGET_S)
    except Exception as e:
        log.warning("perceive failed/timeout: %r", e)
        return None


def _perceive(images: list[bytes], transcript):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types

    t0 = time.time()
    props = {k: types.Schema(type=types.Type.NUMBER, nullable=True) for k in REGIONS}
    props["scene"] = types.Schema(type=types.Type.STRING)
    schema = types.Schema(type=types.Type.OBJECT, properties=props, required=list(REGIONS) + ["scene"])
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
    parts.append(f"{len(images)} frame(s) from the glasses (first/middle/last of a ~3 s clip)."
                 f"\nTranscript of the last ~15 s of mic audio: {transcript or '(none / silence)'}\nScore it.")
    r, m = _gen(PERCEIVE_MODEL, PERCEIVE_MODELS, parts,
                dict(system_instruction=PERCEIVE_SYSTEM, temperature=0.0, max_output_tokens=1024,
                     response_mime_type="application/json", response_schema=schema),
                PERCEIVE_BUDGET_S, "perceive")
    last = None
    if r is not None:
        try:
            d = json.loads(r.text)
            out = {}
            for k in REGIONS:
                v = d.get(k)
                out[k] = None if v is None else float(min(1.0, max(0.0, v)))
            if not images:
                for k in VISUAL_REGIONS:
                    out[k] = None
            out["scene"] = str(d.get("scene") or "")[:120]
            out["model"] = m
            log.info("perceive %s %.1fs: %s", m, time.time() - t0, out)
            return out
        except Exception as e:
            last = e
    log.warning("perceive gave up: %r", last)
    return None


# ---------------- state store / rot model ----------------
def parse_ts(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt


def fresh_state():
    return {
        "regions": {k: {"h": H0, "n": 0, "mean": 0.0, "m2": 0.0, "armed": True, "last_rot_roast": None}
                    for k in REGIONS},
        "last_ts": None,
        "last_roast_ts": None,
        "captures": [],
    }


class Store:
    def __init__(self, root, commit=lambda: None):
        self.root = Path(root)
        self.thumbs = self.root / "thumbs"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        self.commit = commit
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else fresh_state()
        set_params(self.state.get("params"))

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)
        try:
            self.commit()
        except Exception as e:
            log.warning("volume commit failed: %r", e)

    def reset(self):
        with self.lock:
            params = self.state.get("params")
            self.state = fresh_state()
            if params:
                self.state["params"] = params
            shutil.rmtree(self.thumbs, ignore_errors=True)
            self.thumbs.mkdir(parents=True, exist_ok=True)
            self.save()

    def find(self, capture_id):
        for c in self.state["captures"]:
            if c["capture_id"] == capture_id:
                return c
        return None

    def recompute(self, drop: set[str]):
        """Replay the rot model over stored captures (stored region scores, current params).
        Existing roast texts are kept (Gemini is not re-called); dropped captures are removed."""
        with self.lock:
            old = [c for c in self.state["captures"] if c["capture_id"] not in drop]
            last_roast = self.state.get("last_roast_ts")
            params = self.state.get("params")
            self.state = fresh_state()
            if params:
                self.state["params"] = params
            for c in old:
                scores = (c.get("info") or {}).get("norm")
                self._apply_locked(c["capture_id"], datetime.fromisoformat(c["timestamp"]), scores,
                                   c.get("mode") == "audio", c.get("transcript"), None,
                                   {k: c.get(k) for k in ("glasses_battery", "mode", "info", "timings", "has_thumb")},
                                   existing=c, semantic=c.get("semantic"))
            self.state["last_roast_ts"] = last_roast
            for cid in drop:
                (self.thumbs / f"{cid}.jpg").unlink(missing_ok=True)
            self.save()
            return len(self.state["captures"])

    def apply(self, *a, **kw):
        with self.lock:
            rec = self._apply_locked(*a, **kw)
            self.save()
            return rec

    def _apply_locked(self, capture_id, ts: datetime, scores: dict | None, audio_only: bool, transcript, image,
                      extra, existing=None, semantic=None):
        """Decay + heal + roast decision. Returns capture record."""
        if True:
            st = self.state
            R = st["regions"]
            prev_rot = {k: (st["captures"][-1]["regions"][k]["rot"] if st["captures"] else 1 - H0) for k in REGIONS}
            dt_min = 0.0
            if st["last_ts"]:
                dt_min = max(0.0, (ts - datetime.fromisoformat(st["last_ts"])).total_seconds() / 60)
            out, heal = {}, {}
            for k in REGIONS:
                r = R[k]
                if r["h"] > H_FLOOR:
                    r["h"] = H_FLOOR + (r["h"] - H_FLOOR) * 0.5 ** (dt_min * DECAY_MULT.get(k, 1.0) / HALF_LIFE_MIN)
                z, hg, z_t, z_s = 0.0, 0.0, None, None
                if scores is not None:
                    x = scores[k]
                    n0, mu0 = PRIOR_N, PRIOR_MU.get(k, 0.0)
                    n, mean, m2 = r["n"], r["mean"], r["m2"]
                    mu = (n0 * mu0 + n * mean) / (n0 + n)
                    var = (n0 * PRIOR_SD ** 2 + m2 + (n0 * n / (n0 + n)) * (mean - mu0) ** 2) / (n0 + n)
                    z_t = (x - mu) / max(np.sqrt(var), MIN_SD)
                    z = z_t
                    # Welford update
                    r["n"] = n + 1
                    d = x - mean
                    r["mean"] = mean + d / r["n"]
                    r["m2"] = m2 + d * (x - r["mean"])
                sv = (semantic or {}).get(k)
                if sv is not None:
                    z_s = (sv - SEM_CENTER) * SEM_SCALE
                    z = z_s if z_t is None else W_TRIBE * z_t + (1 - W_TRIBE) * z_s
                if (z_t is not None or z_s is not None) and not (audio_only and k in VISUAL_REGIONS):
                    hg = min(MAX_HEAL, GAIN * max(0.0, z - THETA))
                rot_decayed = 1 - r["h"]
                r["h"] = float(min(1.0, max(0.0, r["h"] + hg)))
                rot = 1 - r["h"]
                heal[k] = rot_decayed - rot
                out[k] = {"rot": round(rot, 4), "activation_z": round(float(z), 3),
                          "delta": round(rot - prev_rot[k], 4), "raw": round(scores[k], 4) if scores else None,
                          "z_tribe": None if z_t is None else round(float(z_t), 3),
                          "z_sem": None if z_s is None else round(float(z_s), 3), "heal": round(heal[k], 4)}
                if rot < REARM_THRESH:
                    r["armed"] = True

            # ---- roast decision ----
            def mins_since(iso):
                return 1e9 if not iso else (ts - datetime.fromisoformat(iso)).total_seconds() / 60

            kind = region = None
            if existing is not None:
                kind, region = existing.get("roast_kind"), existing.get("roast_region")
            elif mins_since(st["last_roast_ts"]) >= GLOBAL_GAP_MIN:
                heals = [(heal[k], k) for k in REGIONS if heal[k] >= HEAL_ROAST_DROP]
                rots = [(out[k]["rot"], k) for k in REGIONS
                        if out[k]["rot"] >= ROT_THRESH and R[k]["armed"]
                        and mins_since(R[k]["last_rot_roast"]) >= REGION_COOLDOWN_MIN]
                if heals:
                    kind, region = "heal", max(heals)[1]
                elif rots:
                    kind, region = "rot", max(rots)[1]
            st["last_ts"] = ts.isoformat()
            roast = model_used = None
            if existing is not None:
                roast, model_used = existing.get("roast"), existing.get("roast_model")
            elif kind:
                local = ts.astimezone(LOCAL_TZ).strftime("%-I:%M %p %A")
                roast, model_used = gemini_roast(kind, region, {k: out[k]["rot"] for k in REGIONS},
                                                 transcript, local, image, (semantic or {}).get("scene"))
            if kind:
                st["last_roast_ts"] = ts.isoformat()
                if kind == "rot":
                    R[region]["armed"] = False
                    R[region]["last_rot_roast"] = ts.isoformat()
            rec = {"capture_id": capture_id, "timestamp": ts.isoformat(), "regions": out, "roast": roast,
                   "roast_kind": kind, "roast_region": region, "transcript": transcript,
                   "has_thumb": image is not None, **extra, "roast_model": model_used, "semantic": semantic}
            if existing is not None:
                rec["has_thumb"] = existing.get("has_thumb", False)
            if image is not None:
                (self.thumbs / f"{capture_id}.jpg").write_bytes(image)
            st["captures"].append(rec)
            st["captures"].sort(key=lambda c: c["timestamp"])
            return rec


# ---------------- HTTP API ----------------
PUBLIC_KEYS = ["capture_id", "timestamp", "regions", "roast", "roast_kind", "roast_region", "transcript"]


def make_api(engine: Engine, store: Store):
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    api = FastAPI(title="BrainRot")

    def base_url(req: Request):
        host = req.headers.get("x-forwarded-host") or req.headers.get("host")
        return f"https://{host}"

    def public(rec, req, timeline=False):
        d = {k: rec.get(k) for k in PUBLIC_KEYS}
        d["regions"] = {k: {kk: v[kk] for kk in ("rot", "activation_z", "delta")} for k, v in rec["regions"].items()}
        d["thumb_url"] = f"{base_url(req)}/thumb/{rec['capture_id']}" if rec.get("has_thumb") else None
        if not timeline:
            d["timings"] = rec.get("timings")
        d["glasses_battery"] = rec.get("glasses_battery")
        d["mode"] = rec.get("mode")
        d["semantic"] = rec.get("semantic")
        return d

    @api.get("/health")
    def health():
        return {"ok": True, "model_loaded": engine.model is not None, "n_captures": len(store.state["captures"]),
                "regions": REGIONS, "half_life_min": HALF_LIFE_MIN}

    @api.get("/debug/state")
    def debug_state():
        return store.state

    @api.post("/admin/params")
    async def params(req: Request):
        body = await req.json()
        with store.lock:
            merged = store.state.get("params") or {}
            for k, v in body.items():
                if isinstance(v, dict):
                    merged.setdefault(k, {}).update(v)
                else:
                    merged[k] = v
            store.state["params"] = merged
            cur = set_params(body)
            store.save()
        return cur

    @api.get("/admin/params")
    def get_params():
        return set_params({})

    @api.post("/admin/backfill_perceive")
    def backfill_perceive(force: int = 0, limit: int = 4):
        """Run Gemini perceive on stored captures (middle-frame thumbnail + transcript)."""
        todo = [c for c in list(store.state["captures"])
                if c.get("mode") != "decay_only" and (force or not c.get("semantic"))]
        # sequential + capped per call: Gemini free tier is 5 requests/min/model
        done = 0
        results = {}
        for c in todo[:limit]:
            p = store.thumbs / f"{c['capture_id']}.jpg"
            imgs = [p.read_bytes()] if (c.get("has_thumb") and p.exists()) else []
            try:
                results[c["capture_id"]] = _perceive(imgs, c.get("transcript"))
            except Exception as e:
                log.warning("backfill %s failed: %r", c["capture_id"], e)
        with store.lock:
            for c in store.state["captures"]:
                if results.get(c["capture_id"]):
                    c["semantic"] = results[c["capture_id"]]
                    done += 1
            store.save()
        return {"ok": True, "todo": len(todo), "done": done, "remaining": len(todo) - done}

    @api.post("/admin/recompute")
    def recompute(drop: str = ""):
        n = store.recompute({d for d in drop.split(",") if d})
        return {"ok": True, "n_captures": n}

    @api.post("/reset")
    def reset():
        store.reset()
        return {"ok": True}

    @api.get("/timeline")
    def timeline(req: Request):
        caps = [public(c, req, timeline=True) for c in store.state["captures"]]
        return {"captures": caps, "regions": REGIONS}

    @api.get("/thumb/{capture_id}")
    def thumb(capture_id: str):
        p = store.thumbs / f"{capture_id}.jpg"
        if not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type="image/jpeg")

    @api.post("/capture")
    def capture(
        req: Request,
        frames: list[UploadFile] | None = File(None),
        fps: int = Form(7),
        audio: UploadFile | None = File(None),
        timestamp: str | None = Form(None),
        capture_id: str | None = Form(None),
        glasses_battery: int | None = Form(None),
    ):
        t0 = time.time()
        capture_id = capture_id or str(uuid.uuid4())
        existing = store.find(capture_id)
        if existing:
            return public(existing, req)
        ts = parse_ts(timestamp)
        frame_bytes = [f.file.read() for f in (frames or []) if f is not None]
        frame_bytes = [b for b in frame_bytes if b]
        audio_bytes = audio.file.read() if audio is not None else None
        audio_name = audio.filename if audio is not None else None
        workdir = Path(tempfile.mkdtemp(prefix=f"cap_{capture_id[:8]}_"))
        timings = {}
        try:
            video, wav, info = build_inputs(workdir, frame_bytes, fps, audio_bytes or None, audio_name)
            timings["assemble"] = round(time.time() - t0, 2)
            scores, transcript, mode, semantic = None, None, "decay_only", None
            if video is not None or wav is not None:
                mode = "video+audio" if (video is not None and wav is not None) else (
                    "video" if video is not None else "audio")
                pfut = {}
                sel = [frame_bytes[0], frame_bytes[len(frame_bytes) // 2], frame_bytes[-1]] if frame_bytes else []

                def _start_perceive(tr):
                    pfut["f"] = perceive_async(sel, tr)

                res = engine.predict(video_path=video, audio_path=wav, on_events=_start_perceive)
                timings["tribe_events"] = round(res["t_events"], 2)
                timings["tribe_predict"] = round(res["t_predict"], 2)
                transcript = res["transcript"]
                tp = time.time()
                semantic = perceive_result(pfut.get("f"))
                timings["perceive_wait"] = round(time.time() - tp, 2)
                if res["n_trs"] > 0:
                    norm, absl = region_scores(res["preds"], res["starts"], info.get("clip_s"))
                    scores = absl if SCORE_MODE == "abs" else norm
                    info["norm"] = {k: round(v, 4) for k, v in norm.items()}
                    info["abs"] = {k: round(v, 4) for k, v in absl.items()}
            image = frame_bytes[len(frame_bytes) // 2] if frame_bytes else None
            t1 = time.time()
            rec = store.apply(capture_id, ts, scores, semantic=semantic, audio_only=(mode == "audio"),
                              transcript=transcript,
                              image=image, extra={"glasses_battery": glasses_battery, "mode": mode, "info": info,
                                                  "timings": timings})
            timings["state_roast"] = round(time.time() - t1, 2)
            timings["total"] = round(time.time() - t0, 2)
            log.info("capture %s mode=%s timings=%s roast=%r", capture_id, mode, timings, rec["roast"])
            return public(rec, req)
        except Exception as e:
            log.exception("capture failed")
            return JSONResponse({"error": f"{type(e).__name__}: {e}"[:1000], "capture_id": capture_id},
                                status_code=500)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return api
