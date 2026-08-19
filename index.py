import mimetypes
import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from server.audio_analyzer import analyze
# .js MIME type
mimetypes.add_type("text/javascript", ".js")

app = Flask(__name__, static_folder=".")

# CORS for a separately hosted frontend
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

CNN_MODEL = None
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
# The served checkpoint
MODEL_FILE = "visgen_cnn.pt"


def _model_path():
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    return path if os.path.exists(path) else None


def _load_cnn():
    global CNN_MODEL
    if CNN_MODEL is not None:
        return CNN_MODEL
    path = _model_path()
    if path is None:
        return None
    try:
        import torch
        from server.model import load_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_checkpoint(path, device)
        model.eval()
        model._device = device
        CNN_MODEL = model
        print(f"Loaded model {os.path.basename(path)} (trunk={getattr(model, 'trunk', 'cnn')})")
        return CNN_MODEL
    except ImportError:
        return None


def _cnn_predict(file_path):
    model = _load_cnn()
    if model is None:
        return None
    import numpy as np
    import torch

    # Embedding trunk: CLAP embeds the audio directly, no spectrogram involved.
    # Only reached if a --trunk embedding checkpoint is loaded; the served model is
    # the from-scratch CNN. Part of the CLAP integration (Claude) - see documentation.
    if getattr(model, "trunk", "cnn") == "embedding":
        from server.clap_embeddings import embed_file
        vec = embed_file(file_path, max_duration=60, cache=False)
        tensor = torch.tensor(vec).unsqueeze(0).to(model._device)
        return model.predict(tensor)

    from server.dataset_builder import audio_to_mel_spectrogram, pad_or_truncate
    S_db = audio_to_mel_spectrogram(file_path)
    S_db = pad_or_truncate(S_db)
    spec = (S_db.astype(np.float32) + 80.0) / 80.0
    tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(model._device)
    return model.predict(tensor)


ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "aac", "ogg", "m4a"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


@app.route("/api/analyze", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        file.save(tmp.name)
        tmp.close()
        cnn_out = _cnn_predict(tmp.name)
        if not cnn_out:
            return jsonify({"error": (
                "Trained AI model unavailable. Train it (python -m server.train) so a "
                "checkpoint exists at trained_models/visgen_cnn.pt, then restart the server."
            )}), 503

        results = analyze(tmp.name)
        results["filename"] = secure_filename(file.filename)
        results["cnn"] = cnn_out
        return jsonify(results)
    except Exception as e:
        msg = str(e).strip() or f"{type(e).__name__}: could not process this audio file"
        return jsonify({"error": msg}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(debug=debug, host=host, port=port)
