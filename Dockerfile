# Container for Hugging Face Spaces or any Docker host - see DEPLOY.md

FROM python:3.11-slim

# ffmpeg - librosa's decoding fallback for MP3s libsndfile rejects
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch - --index-url, not --extra-index-url, or pip pulls ~2.5 GB of CUDA
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Separate layer so dependency installs cache across code edits
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Spaces runs as UID 1000 - runtime writes need a directory that user owns
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /home/user/app
COPY --chown=user:user . .

# Spaces routes to 7860; other hosts set $PORT
ENV PORT=7860
EXPOSE 7860

# One worker keeps the model loaded; the long timeout covers analysis + inference
CMD ["sh", "-c", "gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT} index:app"]
