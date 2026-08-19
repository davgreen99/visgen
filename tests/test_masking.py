"""Guards the per-head label masking. Run: python tests/test_masking.py"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.dataset_builder import IGNORE_CLASS, MOOD_CLASSES, STYLE_CLASSES, _row_targets
from server.model import (ALL_HEADS, AUXILIARY_HEADS, CLASSIFICATION_HEADS, REGRESSION_HEADS,
                          REPORTED_HEADS, build_model, decode)
from server.train import UncertaintyWeightedLoss, class_weights, collate_fn, split_dataset

DEAM_ROW = {"filename": "2.mp3", "valence": 0.6, "arousal": 0.4, "label_source": "deam"}
FULL_ROW = {"filename": "t.mp3", "mood": "calm", "style": "lorenz", "energy": 0.5, "valence": 0.5,
            "arousal": 0.5, "color_temperature": 0.5, "complexity": 0.5, "motion_speed": 0.5,
            "brightness": 0.5, "label_source": "bootstrap"}


def test_deam_row_labels_only_valence_and_arousal():
    t = _row_targets(DEAM_ROW)
    labelled = {k for k in REGRESSION_HEADS if np.isfinite(t[k])}
    assert labelled == {"valence", "arousal"}, labelled
    assert t["style"] == IGNORE_CLASS and t["mood"] == IGNORE_CLASS


def test_full_row_labels_every_head():
    t = _row_targets(FULL_ROW)
    assert all(np.isfinite(t[k]) for k in REGRESSION_HEADS)
    assert t["style"] == STYLE_CLASSES.index("lorenz")
    assert t["mood"] == MOOD_CLASSES.index("calm")


def test_mood_is_trained_but_never_reported():
    """Trained for the trunk, never reported."""
    assert "mood" in ALL_HEADS and "mood" in CLASSIFICATION_HEADS, "auxiliary still means trained"
    assert "mood" in _row_targets(FULL_ROW), "no target means no gradient means no auxiliary task"
    assert "mood" in build_model("cnn")(torch.rand(2, 1, 128, 256)), "head must still compute"

    assert "mood" in AUXILIARY_HEADS and "mood" not in REPORTED_HEADS
    out = build_model("cnn")(torch.rand(1, 1, 128, 256))
    assert "mood" not in decode(out), "auxiliary head leaked into the API response"
    assert set(decode(out)) == set(REPORTED_HEADS)


def test_unknown_class_is_ignored_not_coerced_to_zero():
    """The specific old bug: an unrecognised label must not become class 0."""
    t = _row_targets({**FULL_ROW, "style": None})
    assert t["style"] == IGNORE_CLASS, "missing style silently became a real class"
    assert _row_targets({**FULL_ROW, "style": "nonsense"})["style"] == IGNORE_CLASS


def test_only_labelled_heads_receive_gradient():
    """DEAM rows must not train the seven heads they say nothing about."""
    torch.manual_seed(0)
    model = build_model("cnn")
    criterion = UncertaintyWeightedLoss()
    features, targets = collate_fn([(torch.rand(1, 128, 256), _row_targets(DEAM_ROW))
                                    for _ in range(4)])
    loss, parts = criterion(model(features), targets)
    assert set(parts) == {"valence", "arousal"}, parts

    loss.backward()
    touched = {name.split(".")[-2] for name, p in model.heads.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0}
    assert touched == {"valence", "arousal"}, touched


def test_head_with_no_labels_anywhere_is_skipped_entirely():
    """An unlabelled head must not have its log-variance dragged."""
    criterion = UncertaintyWeightedLoss()
    before = criterion.weights()["style"]
    _, targets = collate_fn([(torch.rand(1, 128, 256), _row_targets(DEAM_ROW)) for _ in range(4)])
    preds = {k: (torch.rand(4, len(STYLE_CLASSES)) if k == "style"
                 else torch.rand(4, len(MOOD_CLASSES)) if k == "mood"
                 else torch.rand(4)) for k in ALL_HEADS}
    loss, _ = criterion(preds, targets)
    loss.backward()
    grad = criterion.log_var["style"].grad
    assert grad is None or float(grad) == 0.0, f"unlabelled head accrued gradient: {grad}"
    assert criterion.weights()["style"] == before


def test_split_is_deterministic():
    """Two checkpoints can only be compared if they saw the same validation rows."""
    data = [(torch.zeros(1), {}) for _ in range(100)]
    a = split_dataset(data, 0.2)[1].indices
    b = split_dataset(data, 0.2)[1].indices
    assert a == b, "split differs between calls — checkpoints are not comparable"
    assert split_dataset(data, 0.2, seed=99)[1].indices != a, "seed had no effect"


def test_class_weights_are_inverse_frequency_and_mean_one():
    rows = [(None, _row_targets({**FULL_ROW, "style": "lorenz"})) for _ in range(50)]
    rows += [(None, _row_targets({**FULL_ROW, "style": "waveform"})) for _ in range(5)]
    w = class_weights(rows, heads=("style",))["style"]
    lorenz, waveform = STYLE_CLASSES.index("lorenz"), STYLE_CLASSES.index("waveform")
    assert w[waveform] > w[lorenz], "rare class was not up-weighted"
    assert abs(float(w[waveform] / w[lorenz]) - 10.0) < 0.01, "not inverse-frequency"
    assert abs(float(w.mean()) - 1.0) < 1e-5, "weights must not rescale the loss"


def test_class_weights_ignore_unlabelled_rows():
    """DEAM rows carry no style; they must not be counted as a class."""
    rows = [(None, _row_targets(DEAM_ROW)) for _ in range(20)]
    rows += [(None, _row_targets({**FULL_ROW, "style": "lorenz"})) for _ in range(4)]
    w = class_weights(rows, heads=("style",))["style"]
    assert torch.isfinite(w).all()
    assert abs(float(w.mean()) - 1.0) < 1e-5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failures += 1
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
