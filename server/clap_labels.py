"""Zero-shot labels via CLAP (Wu et al. 2023) for the heads no corpus annotates.
Not ground truth - rows are marked label_source="clap".

AI assistance (Claude, Anthropic): bringing CLAP into the project was Claude's
suggestion, and this module is the integration it drafted. It also predicted, before
the run, that style would be the weakest of the four heads, for the reason given at
STYLE_PROMPTS below - which is why the class distribution was checked before
retraining rather than after, and the only reason the collapse was caught.
See section 9 of docs/Visgen-documentation.docx.

Requires: pip install laion-clap (see DEPLOY.md part 2).
Usage:    python -m server.clap_labels [--heads ...]
"""

import argparse
import os

import numpy as np
import pandas as pd

from .dataset_builder import AUDIO_DIR, METADATA_PATH, STYLE_CLASSES
from .clap_embeddings import embed_file

# Bipolar heads - (low-anchor prompts, high-anchor prompts)
# These three worked. Each asks CLAP to place a track between two opposites, so the
# anchors normalise each other and the softmax lands on the same 0-1 scale as every
# other head. Measured spread against the old analyser formula: colour temperature
# 0.083 -> 0.339, motion speed 0.093 -> 0.258, complexity 0.140 -> 0.118 (the
# analyser had been saturating at exactly 1.000).
#
# brightness is defined here but was deliberately left out of the labelling run: it
# comes from a measured spectral centroid, which is better evidence than a zero-shot
# text comparison. Pass --heads explicitly rather than relying on the default.
BIPOLAR_PROMPTS = {
    "color_temperature": (
        ["a cold, dark, metallic sounding track", "an icy, blue, clinical recording"],
        ["a warm, golden, sunlit sounding track", "a rich, glowing, amber recording"],
    ),
    "complexity": (
        ["a sparse, minimal track with very few elements", "a simple, repetitive, stripped-back recording"],
        ["a dense, intricate track with many layers", "a busy, complex, detailed arrangement"],
    ),
    "motion_speed": (
        ["a slow, still, motionless track", "a static, drifting, barely-moving recording"],
        ["a fast, driving, rapidly moving track", "an urgent, propulsive, high-speed recording"],
    ),
    "brightness": (
        ["a dark, muffled, bass-heavy track", "a dull, low-passed, murky recording"],
        ["a bright, crisp, treble-forward track", "a sharp, airy, high-frequency recording"],
    ),
}

# Style prompts - argmax gives the style head a zero-shot label.
#
# REVERTED - kept as the record of the one head that failed. Nobody has ever
# captioned audio with descriptions of visual forms, so the similarities come back
# nearly identical and argmax returns the most generic prompt rather than the best
# fit. It assigned "thomas" to 264 of 378 tracks (70% of the library on one
# renderer, against the decision tree's 40). No metric could have caught this: once
# CLAP owns the training labels it owns the validation labels too, so the harness
# would have measured agreement with CLAP rather than correctness, and a 70%
# single-class target is trivially learnable by always answering "thomas". The class
# distribution was the only evidence. style in metadata.csv comes from the decision
# tree; do not re-run this head without re-reading section 6 of the documentation.
STYLE_PROMPTS = {
    "harmonograph": "a gentle, flowing, pendulum-like piece with smooth interlocking curves of sound",
    "chladni": "a resonant, sustained, standing-wave drone with symmetrical harmonic structure",
    "dejong": "a granular, scattered, particulate texture of many small sonic points",
    "waveform": "a simple, rippling, undulating wave of sound with steady pulsation",
    "lorenz": "a turbulent, swirling, chaotic piece that never quite repeats",
    "thomas": "a slow, tangled, labyrinthine piece that winds through itself",
    "aizawa": "an intricate, blooming, spiralling piece with an ornate shell of detail",
}


def _text_embeddings(model, prompts):
    vecs = model.get_text_embedding(list(prompts), use_tensor=False)
    return np.asarray(vecs, dtype=np.float32)


def _cosine(audio, text):
    a = audio / (np.linalg.norm(audio) + 1e-9)
    t = text / (np.linalg.norm(text, axis=-1, keepdims=True) + 1e-9)
    return t @ a


def _softmax(x, temperature=0.05):
    z = np.asarray(x, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def score_track(audio_vec, text_cache, heads):
    """Zero-shot scores for one track's embedding."""
    out = {}
    for head in heads:
        if head == "style":
            sims = _cosine(audio_vec, text_cache["style"])
            out["style"] = STYLE_CLASSES[int(np.argmax(sims))]
            continue
        low, high = text_cache[head]
        pair = [float(_cosine(audio_vec, low).mean()), float(_cosine(audio_vec, high).mean())]
        out[head] = round(float(_softmax(pair)[1]), 3)
    return out


def _build_text_cache(model, heads):
    cache = {}
    for head in heads:
        if head == "style":
            cache["style"] = _text_embeddings(model, [STYLE_PROMPTS[s] for s in STYLE_CLASSES])
        else:
            low, high = BIPOLAR_PROMPTS[head]
            cache[head] = (_text_embeddings(model, low), _text_embeddings(model, high))
    return cache


def label(heads=None, audio_dir=None, metadata_path=METADATA_PATH, max_duration=60,
          overwrite=False):
    """Score every dataset track and write the results into metadata.csv."""
    heads = list(heads or (list(BIPOLAR_PROMPTS) + ["style"]))
    unknown = [h for h in heads if h != "style" and h not in BIPOLAR_PROMPTS]
    if unknown:
        raise ValueError(f"no prompts defined for {unknown}")

    from .clap_embeddings import _load
    model = _load()
    text_cache = _build_text_cache(model, heads)

    audio_dir = audio_dir or AUDIO_DIR
    metadata = pd.read_csv(metadata_path)
    for column in heads + ["label_source"]:
        if column not in metadata.columns:
            metadata[column] = np.nan

    scored = 0
    for i, row in metadata.iterrows():
        name = str(row["filename"])
        path = os.path.join(audio_dir, name)
        if not os.path.exists(path):
            continue
        if not overwrite and str(row.get("label_source")) == "deam":
            continue
        try:
            vec = embed_file(path, max_duration=max_duration, cache_name=name)
        except Exception as exc:
            safe = name.encode("ascii", "replace").decode("ascii")
            print(f"  skip ({safe}): {exc}", flush=True)
            continue

        for head, value in score_track(vec, text_cache, heads).items():
            metadata.at[i, head] = value
        metadata.at[i, "label_source"] = "clap"
        scored += 1
        if scored % 25 == 0:
            print(f"  {scored} scored...", flush=True)

    metadata.to_csv(metadata_path, index=False)
    print(f"Scored {scored} tracks on {', '.join(heads)} -> {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--heads", nargs="*", default=None,
                        help=f"subset of {sorted(list(BIPOLAR_PROMPTS) + ['style'])}")
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument("--overwrite", action="store_true",
                        help="also overwrite rows carrying human (DEAM) annotations")
    args = parser.parse_args()
    label(heads=args.heads, audio_dir=args.audio_dir, max_duration=args.max_duration,
          overwrite=args.overwrite)


if __name__ == "__main__":
    main()
