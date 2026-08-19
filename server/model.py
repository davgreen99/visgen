import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_builder import MOOD_CLASSES, STYLE_CLASSES, N_MELS, FIXED_LENGTH

# Every head the model predicts, split by loss type
CLASSIFICATION_HEADS = ("mood", "style")
REGRESSION_HEADS = ("energy", "valence", "arousal", "color_temperature",
                    "complexity", "motion_speed", "brightness")
ALL_HEADS = CLASSIFICATION_HEADS + REGRESSION_HEADS

# Heads trained but never reported - auxiliary signal for the trunk (Caruana 1997)
AUXILIARY_HEADS = ("mood",)
REPORTED_HEADS = tuple(h for h in ALL_HEADS if h not in AUXILIARY_HEADS)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class MultiTaskHeads(nn.Module):
    """The nine output heads, shared by every trunk."""

    def __init__(self, in_dim, n_moods=len(MOOD_CLASSES), n_styles=len(STYLE_CLASSES)):
        super().__init__()
        self.mood = nn.Linear(in_dim, n_moods)
        self.style = nn.Linear(in_dim, n_styles)
        self.regression = nn.ModuleDict({k: nn.Linear(in_dim, 1) for k in REGRESSION_HEADS})

    def forward(self, shared):
        out = {"mood": self.mood(shared), "style": self.style(shared)}
        for key, layer in self.regression.items():
            out[key] = torch.sigmoid(layer(shared)).squeeze(-1)
        return out


def decode(out):
    """One forward pass into the JSON the renderer consumes."""
    result = {}
    for key in REPORTED_HEADS:
        if key in CLASSIFICATION_HEADS:
            classes = MOOD_CLASSES if key == "mood" else STYLE_CLASSES
            result[key] = classes[out[key].argmax(dim=-1).item()]
        else:
            result[key] = round(out[key].item(), 3)
    return result


class VisgenCNN(nn.Module):
    """Multi-task CNN: (batch, 1, 128, 256) mel spectrogram in, head dict out."""

    trunk = "cnn"

    def __init__(self, n_moods=len(MOOD_CLASSES), n_styles=len(STYLE_CLASSES)):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.heads = MultiTaskHeads(128, n_moods, n_styles)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.heads(self.shared_fc(x))

    def predict(self, spectrogram_tensor):
        self.eval()
        with torch.no_grad():
            return decode(self.forward(spectrogram_tensor))


class VisgenEmbeddingNet(nn.Module):
    """Multi-task heads over a frozen pretrained embedding. Alternative trunk.

    Added alongside the CLAP integration (Claude, Anthropic) so the same nine heads
    could be fitted on CLAP's embeddings instead of the from-scratch trunk. Kept as a
    comparison, never served - see VisgenCNN above and section 9 of the documentation.
    """

    trunk = "embedding"

    def __init__(self, in_dim, hidden=256, n_moods=len(MOOD_CLASSES), n_styles=len(STYLE_CLASSES)):
        super().__init__()
        self.in_dim = in_dim
        self.shared_fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )
        self.heads = MultiTaskHeads(128, n_moods, n_styles)

    def forward(self, x):
        return self.heads(self.shared_fc(x))

    def predict(self, embedding_tensor):
        self.eval()
        with torch.no_grad():
            return decode(self.forward(embedding_tensor))


def build_model(trunk="cnn", in_dim=None):
    """Construct whichever trunk a checkpoint or CLI flag asks for."""
    if trunk == "cnn":
        return VisgenCNN()
    if trunk == "embedding":
        if not in_dim:
            raise ValueError("the embedding trunk needs in_dim (the backbone's output width)")
        return VisgenEmbeddingNet(in_dim)
    raise ValueError(f"unknown trunk {trunk!r} (expected 'cnn' or 'embedding')")


def save_checkpoint(model, path, **meta):
    """Save a checkpoint with its trunk and embedding width."""
    torch.save({
        "trunk": getattr(model, "trunk", "cnn"),
        "in_dim": getattr(model, "in_dim", None),
        "state_dict": model.state_dict(),
        **meta,
    }, path)


def load_checkpoint(path, device):
    """Load either checkpoint format; bare state_dicts (pre-refactor) load as VisgenCNN."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        model = VisgenCNN()
        model.load_state_dict(_drop_retired_heads(_upgrade_legacy_keys(blob), model))
        return model.to(device), {"trunk": "cnn", "backend": None}

    model = build_model(blob.get("trunk", "cnn"), blob.get("in_dim"))
    model.load_state_dict(_drop_retired_heads(blob["state_dict"], model))
    meta = {k: v for k, v in blob.items() if k != "state_dict"}
    return model.to(device), meta


def _drop_retired_heads(state, model):
    """Drop weights for heads the model no longer has."""
    known = set(model.state_dict())
    retired = [k for k in state if k not in known]
    if retired:
        heads = sorted({k.split(".")[1] for k in retired if k.startswith("heads.")})
        print(f"  checkpoint carries retired head(s) {heads or retired}: ignoring those weights")
    return {k: v for k, v in state.items() if k in known}


def _upgrade_legacy_keys(state):
    """Map the pre-MultiTaskHeads parameter names onto the current module layout."""
    renames = {"mood_head": "heads.mood", "style_head": "heads.style",
               "color_temp_head": "heads.regression.color_temperature",
               "motion_head": "heads.regression.motion_speed"}
    for key in ("energy", "valence", "arousal", "complexity", "brightness"):
        renames[f"{key}_head"] = f"heads.regression.{key}"

    out = {}
    for key, value in state.items():
        prefix = key.rsplit(".", 1)[0]
        out[key.replace(prefix, renames[prefix], 1) if prefix in renames else key] = value
    return out
