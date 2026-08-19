import subprocess

import librosa
import numpy as np


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_audio(path, sr=22050, mono=True, duration=None):
    """Load audio via librosa, with an ffmpeg fallback for MP3s libsndfile rejects."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=mono, duration=duration)
        if y.size:
            return y, sr
    except Exception:
        pass
    return _ffmpeg_load(path, sr=sr, mono=mono, duration=duration)


def _ffmpeg_load(path, sr=22050, mono=True, duration=None):
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "Could not decode this audio file and no ffmpeg fallback is available. "
            "Install it with:  pip install imageio-ffmpeg  (or install system ffmpeg), "
            "or convert the track to WAV."
        )
    channels = 1 if mono else 2
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "f32le", "-ac", str(channels), "-ar", str(sr), "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        raise RuntimeError("Could not decode audio: " + (detail[-1] if detail else "unknown ffmpeg error"))
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if not mono:
        y = y.reshape(-1, 2).T
    return y, sr


def analyze(file_path, sr=22050, max_duration=None):
    y, sr = load_audio(file_path, sr=sr, duration=max_duration)
    duration = librosa.get_duration(y=y, sr=sr)

    results = {
        "duration": round(duration, 2),
        "sample_rate": sr,
    }

    results.update(_tempo_and_beats(y, sr))
    results.update(_key_detection(y, sr))
    results.update(_spectral_features(y, sr))
    results.update(_energy(y))
    results.update(_timbre(y, sr))
    results.update(_mood(y, sr, results))
    results.update(_structure(y, sr, results))

    return results


def _tempo_and_beats(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_dynamic = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=None
    )

    return {
        "bpm": round(float(np.atleast_1d(tempo)[0]), 1),
        "beat_count": len(beat_times),
        "beat_times": beat_times,
        "tempo_stability": round(
            1.0 - float(np.std(tempo_dynamic) / (np.mean(tempo_dynamic) + 1e-6)), 3
        ),
    }


def _key_detection(y, sr):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_corrs = np.array([
        np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        for i in range(12)
    ])
    minor_corrs = np.array([
        np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
        for i in range(12)
    ])

    if np.max(major_corrs) >= np.max(minor_corrs):
        key_idx = int(np.argmax(major_corrs))
        mode = "major"
        confidence = float(np.max(major_corrs))
    else:
        key_idx = int(np.argmax(minor_corrs))
        mode = "minor"
        confidence = float(np.max(minor_corrs))

    return {
        "key": f"{KEY_NAMES[key_idx]} {mode}",
        "key_confidence": round(confidence, 3),
        "chroma_profile": chroma_mean.tolist(),
    }


def _spectral_features(y, sr):
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    flatness = float(np.mean(spec_flat))

    return {
        "spectral_centroid_mean": round(float(np.mean(spec_cent)), 1),
        "spectral_bandwidth_mean": round(float(np.mean(spec_bw)), 1),
        "spectral_rolloff_mean": round(float(np.mean(spec_rolloff)), 1),
        "spectral_contrast_mean": np.mean(spec_contrast, axis=1).round(2).tolist(),
        "spectral_flatness_mean": round(flatness, 6),
        "angularity": _angularity(flatness, float(np.mean(spec_rolloff)), sr),
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 5),
        "spectral_centroid_timeline": _downsample(spec_cent, 200),
        "spectral_bandwidth_timeline": _downsample(spec_bw, 200),
    }


def _angularity(flatness, rolloff, sr):
    """Spectral sharpness on 0..1 - the bouba/kiki axis (Ramachandran & Hubbard 2001)."""
    noisiness = (np.log10(max(flatness, 1e-6)) + 4.0) / 3.0
    highness = rolloff / (sr / 2.0)
    return round(float(np.clip(noisiness * 0.75 + highness * 0.25, 0.0, 1.0)), 3)


def _energy(y):
    rms = librosa.feature.rms(y=y)[0]
    return {
        "energy_mean": round(float(np.mean(rms)), 5),
        "energy_max": round(float(np.max(rms)), 5),
        "energy_timeline": _downsample(rms, 200),
    }


def _timbre(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)

    return {
        "mfcc_means": mfcc_means.round(3).tolist(),
        "mfcc_variance": mfcc_vars.round(3).tolist(),
        "timbre_brightness": round(float(mfcc_means[1]), 3),
    }


def _mood(y, sr, partial_results):
    bpm = partial_results.get("bpm", 120)
    energy = partial_results.get("energy_mean", 0.05)
    brightness = partial_results.get("spectral_centroid_mean", 2000)
    mode = "major" if "major" in partial_results.get("key", "") else "minor"

    valence, arousal = _mood_scalars(bpm, energy, brightness, 1.0 if mode == "major" else 0.0)

    if valence > 0.6 and arousal > 0.6:
        mood_label = "Energetic"
    elif valence > 0.6 and arousal <= 0.6:
        mood_label = "Calm / Happy"
    elif valence <= 0.4 and arousal > 0.6:
        mood_label = "Aggressive / Tense"
    elif valence <= 0.4 and arousal <= 0.4:
        mood_label = "Dark / Melancholic"
    else:
        mood_label = "Neutral"

    return {
        "mood": mood_label,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
    }


def visual_scalars(bpm, energy_mean, centroid, zcr, valence, arousal, angularity=0.5):
    """The seven 0..1 visual parameters, shared by every caller."""
    energy_norm = min(energy_mean / 0.15, 1.0)
    brightness_norm = min(centroid / 5000.0, 1.0)
    tempo_norm = min(max((bpm - 60) / 120.0, 0.0), 1.0)

    return {
        "energy": round(energy_norm, 3),
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "brightness": round(brightness_norm, 3),
        "complexity": round(min((zcr * 10 + energy_norm) / 2, 1.0), 3),
        "motion_speed": round((tempo_norm + energy_norm) / 2, 3),
        "color_temperature": round((valence + brightness_norm) / 2, 3),
        "angularity": round(angularity, 3),
    }


def _mood_scalars(bpm, energy_mean, centroid, mode_val):
    """Valence/arousal on the Russell (1980) circumplex, from Juslin & Laukka (2003) cues."""
    energy_norm = min(energy_mean / 0.15, 1.0)
    tempo_norm = min(max((bpm - 60) / 120.0, 0.0), 1.0)
    brightness_norm = min(centroid / 5000.0, 1.0)
    valence = 0.3 * mode_val + 0.25 * tempo_norm + 0.25 * brightness_norm + 0.2 * energy_norm
    arousal = 0.4 * energy_norm + 0.35 * tempo_norm + 0.25 * brightness_norm
    return valence, arousal


def _structure(y, sr, partial_results):
    """Section boundaries via Foote's self-similarity method (Foote 1999, 2000)."""
    duration = len(y) / sr
    bpm = partial_results.get("bpm", 120)
    mode_val = 1.0 if "major" in partial_results.get("key", "") else 0.0

    boundaries = _segment_boundaries(y, sr, duration, partial_results.get("beat_times", []))
    segments = [
        _segment_vector(y, sr, start, end, bpm, mode_val)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    return {"structure": {"segments": segments, "track": _section_reference(segments)}}


def _section_reference(segments):
    """Mean over sections, not the whole-track value."""
    keys = ["energy", "valence", "arousal", "brightness", "complexity",
            "motion_speed", "color_temperature", "angularity"]
    if not segments:
        return {k: 0.5 for k in keys}
    return {k: round(float(np.mean([s.get(k, 0.5) for s in segments])), 3) for k in keys}


MIN_SEGMENT_SEC = 8.0
MAX_SEGMENTS = 14


def _segment_boundaries(y, sr, duration, beat_times=()):
    """Section boundary times in seconds, always including 0 and the track end."""
    if duration < MIN_SEGMENT_SEC * 2:
        return [0.0, duration]

    hop = 512
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
        beat_frames = librosa.time_to_frames(np.asarray(beat_times), sr=sr, hop_length=hop)
        beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < mfcc.shape[1])]

        if len(beat_frames) >= 16:
            feat = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
            times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        else:
            feat, times = mfcc, librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)

        if feat.shape[1] < 24:
            return [0.0, duration]

        feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-6)

        ssm = librosa.segment.recurrence_matrix(feat, mode="affinity", width=3, sym=True)
        novelty = _foote_novelty(ssm)

        peaks = librosa.util.peak_pick(
            novelty, pre_max=8, post_max=8, pre_avg=12, post_avg=12,
            delta=float(np.std(novelty) * 0.4), wait=8,
        )
        cand = sorted((float(times[p]), float(novelty[p])) for p in peaks if p < len(times))
    except Exception:
        return [0.0, duration]

    if len(cand) < 2:
        cand = [(t, 1.0) for t in _agglomerative_boundaries(feat, times, duration)]

    return _enforce_min_length(cand, duration)


def _foote_novelty(ssm):
    """Foote's checkerboard-kernel novelty, slid along the SSM diagonal."""
    n = ssm.shape[0]
    half = int(np.clip(n // 10, 4, 32))
    grid = np.arange(-half, half + 1)
    taper = np.exp(-0.5 * (grid / (half / 2.0 + 1e-6)) ** 2)
    kernel = np.outer(taper, taper) * np.sign(np.outer(grid, grid))

    padded = np.pad(ssm, half, mode="edge")
    novelty = np.array([
        float((padded[i:i + 2 * half + 1, i:i + 2 * half + 1] * kernel).sum())
        for i in range(n)
    ])
    novelty -= novelty.min()
    peak = novelty.max()
    return novelty / peak if peak > 0 else novelty


def _agglomerative_boundaries(feat, times, duration):
    """Fallback when the novelty curve is too flat to peak-pick (very uniform tracks)."""
    try:
        k = int(np.clip(round(duration / 30.0), 2, MAX_SEGMENTS))
        idx = librosa.segment.agglomerative(feat, k)
        return sorted(float(times[i]) for i in idx if i < len(times))
    except Exception:
        return []


def _enforce_min_length(candidates, duration):
    """Final section edges: enforce MIN_SEGMENT_SEC and cap at MAX_SEGMENTS."""
    kept = []
    last = 0.0
    for t, strength in candidates:
        if t - last >= MIN_SEGMENT_SEC and duration - t >= MIN_SEGMENT_SEC:
            kept.append((t, strength))
            last = t

    if len(kept) > MAX_SEGMENTS - 1:
        strongest = sorted(kept, key=lambda ts: ts[1], reverse=True)[:MAX_SEGMENTS - 1]
        kept = sorted(strongest)

    return [0.0] + [t for t, _ in kept] + [float(duration)]


def _segment_vector(y, sr, start, end, bpm, mode_val):
    """One section's own visual scalars, from that slice of audio alone."""
    lo, hi = int(start * sr), min(int(end * sr), len(y))
    seg = y[lo:hi]
    if seg.size < sr // 4:
        seg = y[lo:min(lo + sr, len(y))]

    energy = float(np.mean(librosa.feature.rms(y=seg)[0]))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)[0]))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr)[0]))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg)[0]))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(seg)[0]))

    valence, arousal = _mood_scalars(bpm, energy, centroid, mode_val)
    vec = visual_scalars(bpm, energy, centroid, zcr, valence, arousal,
                         _angularity(flatness, rolloff, sr))
    vec["start"] = round(float(start), 2)
    vec["end"] = round(float(end), 2)
    return vec


def _downsample(arr, target_len):
    if len(arr) <= target_len:
        return [round(float(v), 4) for v in arr]
    step = len(arr) / target_len
    return [round(float(arr[int(i * step)]), 4) for i in range(target_len)]
