import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .dataset_builder import (IGNORE_CLASS, MOOD_CLASSES, STYLE_CLASSES, EmbeddingDataset,
                              SpectrogramDataset, build_spectrograms)
from .model import (ALL_HEADS, CLASSIFICATION_HEADS, REGRESSION_HEADS, build_model,
                    save_checkpoint)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models")

CLASSIFICATION_KEYS = set(CLASSIFICATION_HEADS)
REGRESSION_KEYS = set(REGRESSION_HEADS)

# Seeded split, evaluate.py rebuilds the same one
SPLIT_SEED = 1234


def split_dataset(dataset, val_split=0.2, seed=SPLIT_SEED):
    """Deterministic train/val split, shared by the trainer and the evaluator."""
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    targets = {}
    for key in batch[0][1]:
        vals = [item[1][key] for item in batch]
        dtype = torch.long if key in CLASSIFICATION_KEYS else torch.float32
        targets[key] = torch.tensor(vals, dtype=dtype)
    return features, targets


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with a learned log-variance per head (Kendall, Gal & Cipolla 2018)."""

    def __init__(self, heads=ALL_HEADS, class_weights=None):
        super().__init__()
        self.log_var = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in heads})
        self.class_weights = set()
        for key, weight in (class_weights or {}).items():
            self.register_buffer(f"cw_{key}", weight)
            self.class_weights.add(key)

    def forward(self, preds, targets):
        total = torch.zeros((), device=next(self.parameters()).device)
        parts, active = {}, 0

        for key in self.log_var:
            if key not in preds or key not in targets:
                continue
            target = targets[key]
            s = self.log_var[key]

            if key in CLASSIFICATION_KEYS:
                if not (target != IGNORE_CLASS).any():
                    continue
                weight = getattr(self, f"cw_{key}", None) if key in self.class_weights else None
                raw = F.cross_entropy(preds[key], target, ignore_index=IGNORE_CLASS, weight=weight)
                weighted = torch.exp(-s) * raw + 0.5 * s
            else:
                mask = torch.isfinite(target)
                if not mask.any():
                    continue
                raw = F.mse_loss(preds[key][mask], target[mask])
                weighted = 0.5 * torch.exp(-s) * raw + 0.5 * s

            total = total + weighted
            parts[key] = float(raw.detach())
            active += 1

        if not active:
            total = total + 0.0 * sum(p.sum() for p in self.parameters())
        return total, parts

    def weights(self):
        """Effective per-head weight e^(−s), for reporting."""
        return {k: float(torch.exp(-v.detach())) for k, v in self.log_var.items()}


def unweighted_loss(preds, targets):
    """Plain summed task loss, used for checkpoint selection."""
    total, counted = 0.0, 0
    with torch.no_grad():
        for key in CLASSIFICATION_KEYS:
            if key in preds and (targets[key] != IGNORE_CLASS).any():
                total += float(F.cross_entropy(preds[key], targets[key], ignore_index=IGNORE_CLASS))
                counted += 1
        for key in REGRESSION_KEYS:
            if key not in preds:
                continue
            mask = torch.isfinite(targets[key])
            if mask.any():
                total += float(F.mse_loss(preds[key][mask], targets[key][mask]))
                counted += 1
    return total / max(counted, 1)


def spec_augment(batch, freq_masks=2, freq_width=24, time_masks=2, time_width=40, fill=0.0):
    """SpecAugment (Park et al. 2019): erase random frequency bands and time spans.
    Applied in the training loop, not the datasets' transform slot."""
    batch = batch.clone()
    n_mels, n_frames = batch.shape[-2], batch.shape[-1]

    for i in range(batch.shape[0]):
        for _ in range(freq_masks):
            width = int(torch.randint(0, freq_width + 1, (1,)))
            if width:
                start = int(torch.randint(0, max(1, n_mels - width), (1,)))
                batch[i, :, start:start + width, :] = fill
        for _ in range(time_masks):
            width = int(torch.randint(0, time_width + 1, (1,)))
            if width:
                start = int(torch.randint(0, max(1, n_frames - width), (1,)))
                batch[i, :, :, start:start + width] = fill
    return batch


def class_weights(train_set, heads=CLASSIFICATION_HEADS):
    """Inverse-frequency class weights, counted over the training split only."""
    counts = {k: {} for k in heads}
    for _, targets in train_set:
        for key in heads:
            label = targets[key]
            if label != IGNORE_CLASS:
                counts[key][label] = counts[key].get(label, 0) + 1

    sizes = {"mood": len(MOOD_CLASSES), "style": len(STYLE_CLASSES)}
    out = {}
    for key in heads:
        if not counts[key]:
            continue
        n = sizes[key]
        raw = torch.tensor([1.0 / counts[key][c] if counts[key].get(c) else 1.0 for c in range(n)])
        out[key] = raw / raw.mean()
    return out


def _build_dataset(trunk):
    if trunk == "cnn":
        print("Building spectrograms...")
        build_spectrograms()
        return SpectrogramDataset(), None

    dataset = EmbeddingDataset()
    if len(dataset) == 0:
        return dataset, None
    return dataset, int(dataset[0][0].shape[-1])


def _no_data_message(trunk):
    print("\nNo training data found.")
    print("To populate the dataset:")
    print("  1. Place audio files in dataset/audio/")
    print("  2. Run: python -m server.dataset_builder bootstrap")
    if trunk == "cnn":
        print("  3. Run: python -m server.dataset_builder")
        print("  4. Run: python -m server.train")
    else:
        print("  3. Run: python -m server.clap_embeddings")
        print("  4. Run: python -m server.train --trunk embedding")


def train(epochs=50, batch_size=16, lr=1e-3, val_split=0.2, trunk="cnn",
          out_name=None, augment=False, balance_classes=False):
    print(f"Device: {DEVICE} | trunk: {trunk}")

    if augment and trunk != "cnn":
        print("SpecAugment ignored: it needs spectrograms, and this trunk trains on embeddings.")
        augment = False
    if augment:
        print("SpecAugment: on (training batches only)")

    dataset, in_dim = _build_dataset(trunk)
    if len(dataset) == 0:
        _no_data_message(trunk)
        return

    train_set, val_set = split_dataset(dataset, val_split)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)

    weights = class_weights(train_set) if balance_classes else {}
    for key, w in weights.items():
        names = MOOD_CLASSES if key == "mood" else STYLE_CLASSES
        spread = ", ".join(f"{n}={v:.2f}" for n, v in zip(names, w.tolist()))
        print(f"Class weights ({key}): {spread}")

    model = build_model(trunk, in_dim).to(DEVICE)
    criterion = UncertaintyWeightedLoss(class_weights=weights).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, out_name or ("visgen_cnn.pt" if trunk == "cnn" else "visgen_clap.pt"))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(DEVICE)
            if augment:
                features = spec_augment(features)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            loss, _ = criterion(model(features), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_weighted = val_plain = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                preds = model(features)
                loss, _ = criterion(preds, targets)
                val_weighted += loss.item()
                val_plain += unweighted_loss(preds, targets)

        avg_weighted = val_weighted / len(val_loader)
        avg_plain = val_plain / len(val_loader)
        scheduler.step(avg_plain)

        print(f"Epoch {epoch:3d}/{epochs} | train: {avg_train:.4f} | val: {avg_weighted:.4f} "
              f"| val(unweighted): {avg_plain:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}",
              flush=True)

        if avg_plain < best_val:
            best_val = avg_plain
            save_checkpoint(model, path, val_loss=avg_plain, epoch=epoch,
                            split_seed=SPLIT_SEED, val_split=val_split, spec_augment=augment,
                            balance_classes=balance_classes)
            print(f"  -> saved best model (val_loss={avg_plain:.4f})")

    weights = criterion.weights()
    print("\nLearned head weights e^(-s) - a low value means the model found that head hard:")
    for key in sorted(weights, key=weights.get, reverse=True):
        print(f"  {key:<18} {weights[key]:.3f}")
    print(f"\nTraining complete. Best unweighted val loss: {best_val:.4f}")
    print(f"Model saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Visgen multi-task model.")
    parser.add_argument("--trunk", choices=["cnn", "embedding"], default="cnn",
                        help="'cnn' trains the mel-spectrogram network from scratch; "
                             "'embedding' fits heads on a frozen pretrained backbone "
                             "(strongly preferred on a small dataset)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out", default=None, help="checkpoint filename inside trained_models/")
    parser.add_argument("--spec-augment", action="store_true",
                        help="mask random time/frequency stripes in training batches "
                             "(Park et al. 2019); cnn trunk only")
    parser.add_argument("--balance-classes", action="store_true",
                        help="inverse-frequency class weights for mood/style, counted on the "
                             "training split (counters the 152-lorenz-vs-8-waveform skew)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          val_split=args.val_split, trunk=args.trunk, out_name=args.out,
          augment=args.spec_augment, balance_classes=args.balance_classes)


if __name__ == "__main__":
    main()
