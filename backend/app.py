"""BrainRot GPU backend: TRIBE v2 brain-encoding -> region rot model -> Gemini roasts.

Deploy:   modal deploy backend/app.py
Warm:     modal run backend/app.py::warm --n 1      (0 to release)
Download: modal run backend/app.py::download          (one-off weight cache)
"""
import os
from pathlib import Path

import modal

APP_NAME = os.environ.get("BRAINROT_APP", "brainrot")  # use "brainrot-dev" for testing
GPU = os.environ.get("BRAINROT_GPU", "H100")
MIN_CONTAINERS = int(os.environ.get("BRAINROT_WARM", "0"))

HERE = Path(__file__).resolve().parent
MODELS_DIR = "/models"
STATE_DIR = "/state"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "tribev2 @ git+https://github.com/facebookresearch/tribev2.git",
        "faster-whisper>=1.1.0",
        "fastapi[standard]",
        "python-multipart",
        "google-genai",
        "nilearn",
        "pillow",
        "gtts",
    )
    .run_commands("python -m spacy download en_core_web_lg")
    .env(
        {
            "HF_HOME": f"{MODELS_DIR}/hf",
            "TORCH_HOME": f"{MODELS_DIR}/torch",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib",
        }
    )
    .add_local_file(HERE / "region_map.json", "/root/region_map.json")
    .add_local_file(HERE / "brain.py", "/root/brain.py")
)

app = modal.App(APP_NAME, image=image)
models_vol = modal.Volume.from_name("brainrot-models", create_if_missing=True)
state_vol = modal.Volume.from_name(f"{APP_NAME}-state", create_if_missing=True)
# Flags let us build/run before all secrets exist; they're re-injected into the container via a
# dict secret so the remote side resolves the same dependency list.
_SKIP = os.environ.get("BRAINROT_SKIP_SECRETS") == "1"
_NO_GEM = _SKIP or os.environ.get("BRAINROT_NO_GEMINI") == "1"
_flags = {k: "1" for k, on in (("BRAINROT_SKIP_SECRETS", _SKIP), ("BRAINROT_NO_GEMINI", _NO_GEM)) if on}
_extra = [modal.Secret.from_dict(_flags)] if _flags else []
HF_SECRETS = ([] if _SKIP else [modal.Secret.from_name("brainrot-hf")]) + _extra
SECRETS = HF_SECRETS + ([] if _NO_GEM else [modal.Secret.from_name("brainrot-gemini")])


@app.function(gpu=GPU, timeout=600)
def build_check():
    """Builds the image and checks imports (no secrets needed)."""
    import subprocess

    import torch
    import faster_whisper  # noqa: F401
    import tribev2.demo_utils  # noqa: F401
    import brain

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "regions", brain.REGIONS)
    print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])


@app.function(
    gpu=GPU, volumes={MODELS_DIR: models_vol}, secrets=HF_SECRETS, timeout=3600
)
def download():
    """Download all weights (TRIBE + Llama + V-JEPA2 + w2v-bert + whisper) into the volume."""
    import brain

    eng = brain.Engine()
    try:
        eng.load()
        eng.warmup()
        print("download+warmup OK")
    finally:
        models_vol.commit()


@app.cls(
    gpu=GPU,
    volumes={MODELS_DIR: models_vol, STATE_DIR: state_vol},
    secrets=SECRETS,
    cpu=8.0,
    memory=32768,
    timeout=900,
    scaledown_window=1200,
    min_containers=MIN_CONTAINERS,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
class BrainRot:
    @modal.enter()
    def setup(self):
        import brain

        self.engine = brain.Engine()
        self.engine.load()
        try:
            self.engine.warmup()
        except Exception as e:  # never die on warmup
            print("warmup failed:", repr(e))
        try:
            models_vol.commit()  # persist any newly downloaded weights
        except Exception as e:
            print("models commit failed:", repr(e))
        self.store = brain.Store(STATE_DIR, commit=state_vol.commit)

    @modal.asgi_app()
    def web(self):
        import brain

        return brain.make_api(self.engine, self.store)


@app.local_entrypoint()
def warm(n: int = 1):
    cls = modal.Cls.from_name(APP_NAME, "BrainRot", environment_name="brainrot")
    cls().update_autoscaler(min_containers=n)
    print(f"min_containers set to {n}")
