"""CLAP audio embeddings, disk-cached to dataset/embeddings/clap/<name>.npy.

Feeds server/clap_labels.py and the `--trunk embedding` alternative.
Requires `pip install laion-clap` in its own Python 3.11 environment - see below.

AI assistance (Claude, Anthropic): part of the CLAP integration Claude proposed and
drafted - see server/clap_labels.py and section 9 of docs/Visgen-documentation.pdf.
The separate Python 3.11 environment is not optional: laion-clap pins NumPy below 2,
which resolves to a release predating Python 3.13, so pip tries to build it from
source. laion-clap also imports torchvision without declaring it.
"""

import importlib
import os

import numpy as np

from .audio_analyzer import load_audio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(PROJECT_ROOT, "dataset", "embeddings")

BACKEND = "clap"
MODULE = "laion_clap"
SAMPLE_RATE = 48000
EMBED_DIM = 512

_MODEL = None


def is_available():
    """Whether the backend imports here."""
    try:
        importlib.import_module(MODULE)
        return True
    except Exception:
        return False


def _load():
    """The CLAP model, instantiated once per process. Weights download on first use."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        module = importlib.import_module(MODULE)
    except ImportError as exc:
        raise RuntimeError(
            f"CLAP isn't installed. Install it with:  pip install {MODULE.replace('_', '-')}\n"
            "It needs its own Python 3.11 environment: laion-clap pins numpy below 2,\n"
            "and that has no wheel for Python 3.13."
        ) from exc
    _MODEL = module.CLAP_Module(enable_fusion=False)
    _MODEL.load_ckpt()
    return _MODEL


def embed_audio(y):
    """One 512-d vector for a mono waveform already at CLAP's sample rate."""
    vec = _load().get_audio_embedding_from_data(x=y[None, :], use_tensor=False)
    return np.asarray(vec).reshape(-1).astype(np.float32)


def embed_file(path, max_duration=None, cache_name=None, cache=True):
    """Embedding for one file, disk-cached when `cache` is set."""
    cached = _cache_path(cache_name or os.path.basename(path)) if cache else None
    if cached and os.path.exists(cached):
        return np.load(cached)

    y, _ = load_audio(path, sr=SAMPLE_RATE, duration=max_duration)
    vec = embed_audio(y)

    if cached:
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        np.save(cached, vec)
    return vec


def _cache_path(name):
    return os.path.join(EMBED_DIR, BACKEND, os.path.splitext(name)[0] + ".npy")


def build_embeddings(audio_dir=None, metadata_path=None, max_duration=60):
    """Fill the cache for every dataset row whose audio is found."""
    import pandas as pd

    from .dataset_builder import AUDIO_DIR, METADATA_PATH

    audio_dir = audio_dir or AUDIO_DIR
    metadata = pd.read_csv(metadata_path or METADATA_PATH)

    built = skipped = 0
    for name in metadata["filename"]:
        path = os.path.join(audio_dir, str(name))
        if not os.path.exists(path):
            skipped += 1
            continue
        if os.path.exists(_cache_path(str(name))):
            continue
        try:
            embed_file(path, max_duration=max_duration, cache_name=str(name))
            built += 1
            if built % 25 == 0:
                print(f"  {built} embedded...", flush=True)
        except Exception as exc:
            safe = str(name).encode("ascii", "replace").decode("ascii")
            print(f"  skip ({safe}): {exc}", flush=True)
            skipped += 1

    print(f"Done -- {built} new embeddings, {skipped} skipped", flush=True)


if __name__ == "__main__":
    import sys

    if not is_available():
        print(f"CLAP not installed here. pip install {MODULE.replace('_', '-')} "
              "in a separate Python 3.11 environment.")
        raise SystemExit(1)
    build_embeddings(audio_dir=sys.argv[1] if len(sys.argv) > 1 else None)
