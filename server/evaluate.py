"""Per-head validation metrics on the seeded held-out split.

Usage: python -m server.evaluate <checkpoint> [<checkpoint> ...] [--confusion]"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .dataset_builder import MOOD_CLASSES, STYLE_CLASSES
from .model import AUXILIARY_HEADS, CLASSIFICATION_HEADS, REGRESSION_HEADS, load_checkpoint
from .train import DEVICE, IGNORE_CLASS, SPLIT_SEED, _build_dataset, collate_fn, split_dataset

CLASS_NAMES = {"mood": MOOD_CLASSES, "style": STYLE_CLASSES}


def _regression_metrics(pred, target):
    """MAE and R² over labelled entries only."""
    mask = torch.isfinite(target)
    n = int(mask.sum())
    if not n:
        return {"support": 0}
    p, t = pred[mask], target[mask]
    mae = float((p - t).abs().mean())
    var = float(((t - t.mean()) ** 2).sum())
    r2 = float("nan") if var == 0 else 1.0 - float(((t - p) ** 2).sum()) / var
    return {"support": n, "mae": mae, "r2": r2, "pred_std": float(p.std()), "true_std": float(t.std())}


def _classification_metrics(logits, target, n_classes):
    """Accuracy, macro-F1 and confusion over labelled entries only."""
    mask = target != IGNORE_CLASS
    n = int(mask.sum())
    if not n:
        return {"support": 0}
    pred = logits[mask].argmax(dim=-1)
    true = target[mask]

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(true.tolist(), pred.tolist()):
        confusion[t, p] += 1

    f1s, present = [], 0
    for c in range(n_classes):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum()) - tp
        fn = float(confusion[c, :].sum()) - tp
        if tp + fn == 0:
            continue
        present += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    return {
        "support": n,
        "accuracy": float((pred == true).float().mean()),
        "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "classes_present": present,
        "confusion": confusion,
    }


def evaluate(checkpoint_path, trunk="cnn", val_split=0.2, batch_size=32,
             seed=SPLIT_SEED):
    """Score one checkpoint on the held-out split. Returns {head: metrics}."""
    dataset, in_dim = _build_dataset(trunk)
    if len(dataset) == 0:
        raise RuntimeError("No dataset — build features first (python -m server.dataset_builder).")

    _, val_set = split_dataset(dataset, val_split, seed)
    loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)

    model, meta = load_checkpoint(checkpoint_path, DEVICE)
    model.eval()

    preds = {k: [] for k in list(REGRESSION_HEADS) + list(CLASSIFICATION_HEADS)}
    targets = {k: [] for k in preds}
    with torch.no_grad():
        for features, batch_targets in loader:
            out = model(features.to(DEVICE))
            for key in preds:
                if key in out and key in batch_targets:
                    preds[key].append(out[key].cpu())
                    targets[key].append(batch_targets[key])

    results = {}
    for key in REGRESSION_HEADS:
        if preds[key]:
            results[key] = _regression_metrics(torch.cat(preds[key]), torch.cat(targets[key]))
    for key in CLASSIFICATION_HEADS:
        if preds[key]:
            results[key] = _classification_metrics(
                torch.cat(preds[key]), torch.cat(targets[key]), len(CLASS_NAMES[key]))

    return results, meta, len(val_set)


def _print_report(path, results, meta, n_val, show_confusion=False):
    name = os.path.basename(path)
    trained_seed = meta.get("split_seed")
    print(f"\n{'=' * 78}\n{name}  —  {n_val} validation rows")
    bits = [f"epoch {meta['epoch']}" if "epoch" in meta else None,
            f"val_loss {meta['val_loss']:.4f}" if "val_loss" in meta else None,
            "spec_augment" if meta.get("spec_augment") else None]
    if any(bits):
        print("  " + " | ".join(b for b in bits if b))
    if trained_seed is None:
        print("  ! No split_seed recorded: trained before the split was seeded, so some of")
        print("    these rows were probably in its TRAINING set. Retrain to compare fairly.")
    elif trained_seed != SPLIT_SEED:
        print(f"  ! Trained against split_seed={trained_seed}, scoring against {SPLIT_SEED}.")
    print(f"{'=' * 78}")

    print(f"{'head':<20}{'support':>8}{'MAE':>9}{'R2':>9}{'pred sd':>9}{'true sd':>9}")
    for key in REGRESSION_HEADS:
        m = results.get(key)
        if not m:
            continue
        if not m["support"]:
            print(f"{key:<20}{0:>8}{'—':>9}{'—':>9}{'—':>9}{'—':>9}")
            continue
        print(f"{key:<20}{m['support']:>8}{m['mae']:>9.4f}{m['r2']:>9.3f}"
              f"{m['pred_std']:>9.3f}{m['true_std']:>9.3f}")

    print(f"\n{'head':<20}{'support':>8}{'accuracy':>10}{'macro-F1':>10}{'classes':>9}")
    for key in CLASSIFICATION_HEADS:
        m = results.get(key)
        label = f"{key} (aux)" if key in AUXILIARY_HEADS else key
        if not m or not m["support"]:
            print(f"{label:<20}{0:>8}{'—':>10}{'—':>10}{'—':>9}")
            continue
        print(f"{label:<20}{m['support']:>8}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}"
              f"{m['classes_present']:>9}")

    if show_confusion:
        for key in CLASSIFICATION_HEADS:
            m = results.get(key)
            if not m or not m["support"]:
                continue
            names = CLASS_NAMES[key]
            print(f"\n  {key} confusion (rows = true, cols = predicted)")
            print("    " + "".join(f"{n[:6]:>8}" for n in names))
            for i, row in enumerate(m["confusion"].tolist()):
                print(f"    {names[i][:10]:<12}" + "".join(f"{v:>8}" for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", nargs="+", help="checkpoint paths to score")
    parser.add_argument("--trunk", choices=["cnn", "embedding"], default="cnn")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--confusion", action="store_true", help="also print confusion matrices")
    args = parser.parse_args()

    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"\nskip (not found): {path}")
            continue
        results, meta, n_val = evaluate(path, trunk=args.trunk,
                                        val_split=args.val_split, seed=args.seed)
        _print_report(path, results, meta, n_val, show_confusion=args.confusion)


if __name__ == "__main__":
    main()
