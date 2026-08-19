"""Ingest the DEAM corpus (Aljanaki, Yang & Soleymani 2017): human valence/arousal.

Usage: python -m server.deam <deam_root> [audio_dir]
"""

import glob
import os

import pandas as pd

from .dataset_builder import METADATA_PATH

ANNOTATION_GLOB = os.path.join(
    "annotations", "annotations averaged per song", "song_level",
    "static_annotations_averaged_songs_*.csv",
)
AUDIO_SUBDIR = "MEMD_audio"
DEAM_SCALE = (1.0, 9.0)


def _normalize(series):
    lo, hi = DEAM_SCALE
    return ((pd.to_numeric(series, errors="coerce") - lo) / (hi - lo)).clip(0.0, 1.0)


def load_annotations(deam_root):
    """Concatenated song-level annotations, normalized to 0..1."""
    paths = sorted(glob.glob(os.path.join(deam_root, ANNOTATION_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"No DEAM annotation CSVs under {os.path.join(deam_root, ANNOTATION_GLOB)}.\n"
            "Check the path, and that the archive was unpacked with its folder names intact."
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [c.strip() for c in frame.columns]
        frames.append(frame)

    ann = pd.concat(frames, ignore_index=True)
    missing = {"song_id", "valence_mean", "arousal_mean"} - set(ann.columns)
    if missing:
        raise ValueError(f"DEAM annotations are missing expected columns: {sorted(missing)}")

    return pd.DataFrame({
        "song_id": ann["song_id"].astype(int),
        "valence": _normalize(ann["valence_mean"]).round(3),
        "arousal": _normalize(ann["arousal_mean"]).round(3),
    }).dropna()


def ingest(deam_root, metadata_path=METADATA_PATH, audio_subdir=AUDIO_SUBDIR, audio_dir=None):
    """Append DEAM rows to metadata.csv; re-running updates existing rows."""
    annotations = load_annotations(deam_root)
    audio_dir = audio_dir or os.path.join(deam_root, audio_subdir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"No DEAM audio directory at {audio_dir}")

    rows = []
    for _, ann in annotations.iterrows():
        name = f"{int(ann['song_id'])}.mp3"
        if not os.path.exists(os.path.join(audio_dir, name)):
            continue
        rows.append({"filename": name, "valence": ann["valence"], "arousal": ann["arousal"],
                     "label_source": "deam"})

    if not rows:
        raise FileNotFoundError(
            f"Found {len(annotations)} annotations but no matching audio in {audio_dir}."
        )

    new = pd.DataFrame(rows)
    existing = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
    if "filename" in existing.columns:
        existing = existing[~existing["filename"].isin(new["filename"])]

    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(metadata_path, index=False)

    print(f"Wrote {len(new)} DEAM rows to {metadata_path} ({len(combined)} rows total).")
    print(f"\nDEAM audio lives at: {audio_dir}")
    print("Build features for those rows, reading the corpus where it sits:")
    print(f'  python -m server.dataset_builder spectrograms "{audio_dir}"')
    print("Then drop the formula's valence/arousal so these two heads train on human labels only:")
    print("  python -m server.dataset_builder clear-bootstrap-emotion")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python -m server.deam <deam_root> [audio_dir]")
        raise SystemExit(1)
    ingest(sys.argv[1], audio_dir=sys.argv[2] if len(sys.argv) > 2 else None)
