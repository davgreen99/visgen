import os
import concurrent.futures

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio_analyzer import analyze, load_audio, visual_scalars

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
SPEC_DIR = os.path.join(DATASET_DIR, "spectrograms")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

MOOD_CLASSES = ["energetic", "calm", "aggressive", "melancholic", "neutral"]

# Where a row's labels came from - blanks are masked per head, so mixing sources is safe
# "clap" arrived with the CLAP integration (Claude's suggestion - see
# server/clap_labels.py). Provenance is the point: a human DEAM rating, a zero-shot
# CLAP score and a formula's own output must stay distinguishable
LABEL_SOURCES = ["bootstrap", "deam", "clap", "manual"]
# The renderer formulas (static/js/formulas.js), mapped 1:1 to the style head
STYLE_CLASSES = ["harmonograph", "chladni", "dejong", "waveform", "lorenz", "thomas", "aizawa"]


def derive_style(energy, arousal, complexity, brightness):
    """Map analysed features onto a STYLE_CLASSES formula."""
    if arousal < 0.35:
        return "waveform" if complexity < 0.5 else "harmonograph"
    if arousal < 0.6:
        return "chladni" if brightness < 0.5 else "dejong"
    if brightness < 0.4:
        return "lorenz"
    if brightness < 0.7:
        return "thomas"
    return "aizawa"

N_MELS = 128
HOP_LENGTH = 512
FIXED_LENGTH = 256


def audio_to_mel_spectrogram(file_path, sr=22050):
    y, sr = load_audio(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def pad_or_truncate(spectrogram, target_length=FIXED_LENGTH):
    if spectrogram.shape[1] >= target_length:
        return spectrogram[:, :target_length]
    pad_width = target_length - spectrogram.shape[1]
    return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)


def build_spectrograms(audio_dir=None, force_rebuild=False):
    """Build a mel spectrogram for every metadata row whose audio is in `audio_dir`."""
    audio_dir = audio_dir or AUDIO_DIR
    os.makedirs(SPEC_DIR, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    built = missing = 0

    for _, row in metadata.iterrows():
        audio_path = os.path.join(audio_dir, row["filename"])
        spec_name = os.path.splitext(row["filename"])[0] + ".npy"
        spec_path = os.path.join(SPEC_DIR, spec_name)

        if os.path.exists(spec_path) and not force_rebuild:
            continue

        if not os.path.exists(audio_path):
            missing += 1
            continue

        try:
            S_db = audio_to_mel_spectrogram(audio_path)
            S_db = pad_or_truncate(S_db)
            np.save(spec_path, S_db)
            built += 1
            if built % 25 == 0:
                print(f"  {built} built...", flush=True)
        except Exception as e:
            safe = row["filename"].encode("ascii", "replace").decode("ascii")
            print(f"  skip (error): {safe}: {e}", flush=True)

    print(f"Done -- {built} spectrograms built from {audio_dir}, "
          f"{missing} rows not found there, {len(metadata)} total entries", flush=True)


def bootstrap_labels(audio_dir=None):
    """Auto-label a directory of audio with the rule-based analyzer, appending rows to
    metadata.csv."""
    audio_dir = audio_dir or AUDIO_DIR
    extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    metadata = pd.read_csv(METADATA_PATH) if os.path.exists(METADATA_PATH) else pd.DataFrame()
    existing = set(metadata["filename"].tolist()) if "filename" in metadata.columns else set()

    new_rows = []
    for fname in sorted(os.listdir(audio_dir)):
        if os.path.splitext(fname)[1].lower() not in extensions:
            continue
        if fname in existing:
            continue

        audio_path = os.path.join(audio_dir, fname)
        safe_name = fname.encode("ascii", "replace").decode("ascii")
        print(f"  analyzing: {safe_name}", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(analyze, audio_path, 22050, 30)
                result = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            print(f"  skip (timeout): {safe_name}", flush=True)
            continue
        except Exception as e:
            print(f"  error ({safe_name}): {e}", flush=True)
            continue

        vec = visual_scalars(
            result.get("bpm", 120),
            result.get("energy_mean", 0.05),
            result.get("spectral_centroid_mean", 2000),
            result.get("zero_crossing_rate_mean", 0.05),
            result.get("valence", 0.5),
            result.get("arousal", 0.5),
            result.get("angularity", 0.5),
        )

        mood_raw = result.get("mood", "Neutral").lower()
        if "energetic" in mood_raw:
            mood = "energetic"
        elif "calm" in mood_raw or "happy" in mood_raw:
            mood = "calm"
        elif "aggressive" in mood_raw or "tense" in mood_raw:
            mood = "aggressive"
        elif "dark" in mood_raw or "melancholic" in mood_raw:
            mood = "melancholic"
        else:
            mood = "neutral"

        style = derive_style(vec["energy"], vec["arousal"], vec["complexity"], vec["brightness"])

        new_rows.append({
            "filename": fname,
            "mood": mood,
            "energy": vec["energy"],
            "valence": vec["valence"],
            "arousal": vec["arousal"],
            "color_temperature": vec["color_temperature"],
            "complexity": vec["complexity"],
            "motion_speed": vec["motion_speed"],
            "style": style,
            "brightness": vec["brightness"],
            "label_source": "bootstrap",
        })
        print(f"  labeled: {safe_name} -> mood={mood}, style={style}, energy={vec['energy']:.2f}", flush=True)

        if len(new_rows) % 50 == 0:
            _save_rows(metadata, new_rows)
            print(f"  [checkpoint] saved {len(new_rows)} new entries", flush=True)

    _save_rows(metadata, new_rows)


def _save_rows(metadata, new_rows):
    if not new_rows:
        print("No new audio files found to label")
        return
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([metadata, new_df], ignore_index=True)
    combined.to_csv(METADATA_PATH, index=False)
    print(f"\nSaved {len(new_rows)} new entries to metadata.csv ({len(combined)} total)", flush=True)


REGRESSION_COLUMNS = ["energy", "valence", "arousal", "color_temperature",
                      "complexity", "motion_speed", "brightness"]

# Sentinels the trainer masks out - NaN for regression, -1 for class labels
IGNORE_CLASS = -1


def _row_targets(row):
    """Per-head targets for one metadata row, with blanks left as ignore sentinels."""
    targets = {
        "mood": MOOD_CLASSES.index(row["mood"]) if row.get("mood") in MOOD_CLASSES else IGNORE_CLASS,
        "style": STYLE_CLASSES.index(row["style"]) if row.get("style") in STYLE_CLASSES else IGNORE_CLASS,
    }
    for key in REGRESSION_COLUMNS:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        targets[key] = value if np.isfinite(value) else float("nan")
    return targets


class _LabelledDataset(Dataset):
    """Shared row bookkeeping for the spectrogram and embedding datasets."""

    def __init__(self, metadata_path, feature_dir, suffix=".npy", transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.feature_dir = feature_dir
        self.transform = transform

        self.valid_indices = [
            i for i, row in self.metadata.iterrows()
            if os.path.exists(os.path.join(feature_dir, os.path.splitext(str(row["filename"]))[0] + suffix))
        ]
        self.suffix = suffix
        self._report()

    def _report(self):
        n = len(self.valid_indices)
        print(f"Dataset: {n} valid samples out of {len(self.metadata)} entries")
        if not n:
            return
        rows = self.metadata.iloc[self.valid_indices]
        if "label_source" in rows.columns:
            counts = rows["label_source"].fillna("bootstrap").value_counts()
            print("  label sources: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        covered = {k: int(pd.to_numeric(rows[k], errors="coerce").notna().sum())
                   for k in REGRESSION_COLUMNS if k in rows.columns}
        print("  labelled per head: " + ", ".join(f"{k}={v}" for k, v in covered.items()))

    def __len__(self):
        return len(self.valid_indices)

    def _path(self, row):
        return os.path.join(self.feature_dir, os.path.splitext(str(row["filename"]))[0] + self.suffix)

    def __getitem__(self, idx):
        row = self.metadata.iloc[self.valid_indices[idx]]
        return self._features(row), _row_targets(row)

    def _features(self, row):
        raise NotImplementedError


class SpectrogramDataset(_LabelledDataset):
    """Mel spectrograms for the from-scratch CNN trunk."""

    def __init__(self, metadata_path=METADATA_PATH, spec_dir=SPEC_DIR, transform=None):
        super().__init__(metadata_path, spec_dir, transform=transform)

    def _features(self, row):
        spec = np.load(self._path(row)).astype(np.float32)
        spec = (spec + 80.0) / 80.0
        spec = torch.tensor(spec).unsqueeze(0)
        return self.transform(spec) if self.transform else spec


class EmbeddingDataset(_LabelledDataset):
    """Cached CLAP embeddings for the embedding trunk (see server/clap_embeddings.py).

    Part of the CLAP integration drafted with Claude (Anthropic). This is the
    alternative trunk only - the served model is the from-scratch CNN, because a
    frozen backbone would demonstrate someone else's network rather than this one.
    """

    def __init__(self, metadata_path=METADATA_PATH, transform=None):
        from .clap_embeddings import BACKEND, EMBED_DIR
        super().__init__(metadata_path, os.path.join(EMBED_DIR, BACKEND), transform=transform)

    def _features(self, row):
        vec = torch.tensor(np.load(self._path(row)).astype(np.float32))
        return self.transform(vec) if self.transform else vec


def clear_bootstrap_emotion(metadata_path=METADATA_PATH, columns=("valence", "arousal")):
    """Blank the formula-derived valence/arousal on bootstrap rows after a DEAM ingest."""
    metadata = pd.read_csv(metadata_path)
    if "label_source" not in metadata.columns:
        raise ValueError("metadata.csv has no label_source column")

    is_bootstrap = metadata["label_source"].fillna("bootstrap") == "bootstrap"
    cleared = {}
    for column in columns:
        if column not in metadata.columns:
            continue
        target = is_bootstrap & metadata[column].notna()
        cleared[column] = int(target.sum())
        metadata.loc[target, column] = np.nan

    metadata.to_csv(metadata_path, index=False)
    print("Cleared on bootstrap rows: " + ", ".join(f"{k}={v}" for k, v in cleared.items()))
    remaining = {c: int(pd.to_numeric(metadata[c], errors="coerce").notna().sum())
                 for c in columns if c in metadata.columns}
    print("Rows still labelled: " + ", ".join(f"{k}={v}" for k, v in remaining.items()))


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "spectrograms"
    directory = sys.argv[2] if len(sys.argv) > 2 else None
    if command == "bootstrap":
        bootstrap_labels(directory)
    elif command == "clear-bootstrap-emotion":
        clear_bootstrap_emotion()
    else:
        if directory is None and command != "spectrograms" and os.path.isdir(command):
            directory = command
        build_spectrograms(directory)
