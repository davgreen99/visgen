---
title: Visgen
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Visgen — AI Music Visualizer

Upload a track, analyse its sonic characteristics, and render it as a live, one‑of‑a‑kind
generative visual driven by tempo, frequency, and mood.

The trained AI model is the core of the project: every track is analysed by the CNN, and
its prediction is what drives the visual. The browser only handles rendering and live playback reactivity; it never fabricates the analysis.

The project has two halves:

- ML pipeline (`*.py`): PyTorch CNN (`VisgenCNN`) predicts visual parameters from a mel spectrogram, Flask server (`index.py`) serves the app and exposes the model‑backed `POST /api/analyze` endpoint.
- Web app (`index.html`, `static/js/`, `static/css/`): the visualizer built on [Three.js](https://threejs.org/) and [math.js](https://mathjs.org/). On upload it sends the track to `/api/analyze` and renders the formula the model chooses. The Web Audio API manages live reactivity — beats, kicks and bloom — while the one‑shot track analysis comes from the model.

---

### Using the app

1. Upload a track (MP3 / WAV / FLAC / AAC / OGG / M4A, up to 50 MB) — it is analysed server-side for BPM,
   key, energy, timbre and structure, and by the CNN for the nine visual parameters.
2. Tune the visual in the controls panel:
   - Formula: the math that generates the shape (see below).
   - Colours: AI‑derived palette or a multi‑option preset.
   - Combine Layers: stack multiple formulas, each editable independently.
   - Audio Reactive: frequency bands drive the motion.
3. Generate and press play: the visual fills the page as an immersive background as well as the output box — this is where you edit your creation.
4. Drag a layer in the output window to reposition it; double-click recentres it.
5. Screenshot: saves a PNG of the current frame; Save: stores the current
   formulas + parameters + a thumbnail to the Gallery below. Click a saved card to see the track and parameters used, or choose "+ Add to visualizer" to restore it.

>  `music-test/` holds a couple of tracks, so the app can be tried if you don't have any music to hand.

![The controls panel and the visual output](docs/screenshot.png)

### Visual formulas

| Formula | Type | Notes |
|---|---|---|
| Harmonograph | 3‑axis curve | woven string‑art net; pendulum frequencies are CNN/music‑driven |
| Chladni | cymatics shader | smooth nodal‑line plate |
| de Jong | attractor point cloud | 90k‑point cloud; constants driven by the music |
| Waveform | cylinder mesh | radius driven by frequency bands |
| Lorenz / Thomas / Aizawa | strange attractors | classic chaotic line traces |

### How the music drives the visual

Four terms move every parameter, each a fraction of that parameter's own range so one rule fits all seven formulas (`REACT` in `static/js/formulas.js`):

| term | source | what it does |
| --- | --- | --- |
| **section** | server‑side structural segmentation | The analyser splits the track into sections using Foote's self‑similarity method, and reports each section's own feature vector. The renderer re‑runs the formula's `seed()` on it and applies the *difference* from the track‑level seed — so a breakdown genuinely re‑forms into a different shape and the drop re‑forms back. |
| **beat** | the analyser's beat grid, phase‑locked to playback position | A sharp impulse on each beat. Kept separate from level because reading loudness alone makes hits land late and soft. |
| **band** | live FFT | Instantly converts audio into spectrum frequency bands. |
| **wander** | — | Gives life to still passages, drifting so the visual keeps moving. |

The colour mapping follows Palmer, Schloss, Xu & Prado‑León (2013), whose finding is that music‑to‑colour associations are driven by emotion rather than formed directly. Hue runs cool‑to‑warm (dark blue → yellow), valence drives lightness, and arousal drives saturation, so a calm track is dimmer and an agitated one is bright. Hue is shared: `color_temperature`
carries 65% of it and valence 35%, because the model predicts valence in too narrow a band to separate tracks on its own.

Shape angularity follows the bouba/kiki correspondence; the analyser's spectral‑flatness feature drives Chladni line width and de Jong folding. Spectral brightness drives vertical extent (Chladni's vertical mode
count, the harmonograph's vertical pendulums).

---

## ML pipeline

The Python side trains **VisgenCNN** (`server/model.py`), a multi‑task CNN that maps a `128 × 256` mel spectrogram to:

- Classification: `style` — the renderer to use: harmonograph / chladni / dejong /
  waveform / lorenz / thomas / aizawa (maps 1:1 onto the renderer's formulas)
- Regression (0–1): energy, valence, arousal, color_temperature, complexity,
  motion_speed, brightness
- Auxiliary: `mood`. It scores macro-F1 0.26, and removing it made seven of the other eight heads worse, so it shapes the shared trunk as an auxiliary task (Caruana 1997). The app shows the analyser's measured mood instead.

Two trunks feed those heads. `VisgenCNN` learns the whole thing from scratch; the `--trunk embedding` model fits the same heads on a frozen pretrained backbone (`clap_embeddings.py`). Due to the size of the dataset, a few hundred tracks are too few to train a 1M‑parameter conv net (Kong et al. 2020, *PANNs*; Won et al. 2020).
It is kept as an alternative, but the from‑scratch CNN is deliberately the model this project serves — a frozen backbone would demonstrate someone else's network, not this one.

### Setup

```bash
pip install -r requirements.txt
```

### Where labels come from

`metadata.csv` carries a `label_source` column, and it matters:

| source | meaning |
| --- | --- |
| `bootstrap` | the rule‑based librosa analyser's own output, recorded as a label. A model trained **only** on these learns the formula rather than the perception it stands in for — distillation with the wrong teacher (Hinton et al. 2015). The formula becomes the ceiling, and no change of architecture lifts it. |
| `deam` | human valence/arousal annotations from the DEAM corpus (Aljanaki et al. 2017). Real supervision — for those two heads only. |
| `clap` | zero‑shot scores from a language‑audio model, for the aesthetic heads no corpus annotates. |
| `manual` | hand‑labelled. |

Rows may leave any column blank. The dataset masks blanks **per head**, so a DEAM row
contributes gradient to valence and arousal and to nothing else, rather than having the
other seven heads trained on invented values.

### How the model was trained

The dataset grew in three stages, and each one changed what the visuals do. The commands
below are the ones that were actually run, in order.

**Stage 1 — bootstrap labels (384 tracks).** The rule-based analyser labelled its own
training set: every one of the nine heads learned from the formula in `audio_analyzer.py`.

```bash
python -m server.dataset_builder bootstrap      # seeds dataset/metadata.csv
python -m server.dataset_builder spectrograms   # 128 x 256 mel arrays
python -m server.train                          # -> trained_models/visgen_cnn.pt
```

What worked well was how the network reproduced the formulas.
At the same time it was limited, and the failures belonged more to the output than the actual code.
Predicted valence varied by a standard deviation of 0.024 across a 384-track library, meaning most tracks
would output very similarly. Since valence drives lightness and part of hue, every track came out roughly the same colour.

**Stage 2 — DEAM (1,802 tracks, human labels).** DEAM (Aljanaki et al. 2017) supervises only valence and arousal, two of the nine heads, so its direct reach is limited. Introducing DEAM took the trunk from 384 tracks of audio to 2,180, a 5.7× increase in what the convolutional layers ever see, and every head reads from the result.

```bash
python -m server.deam <deam_root> [audio_dir]                    # writes label-only rows
python -m server.dataset_builder spectrograms "<deam_root>/MEMD_audio"
python -m server.dataset_builder clear-bootstrap-emotion         # see below
python -m server.train --balance-classes
```
DEAM improved the seven heads it never labels: every head became more discriminating. What they were targeting never changed; DEAM only allowed them to be more precise. In other words, although the corpus supervises just two heads, all nine share one trunk, so the whole output benefits.
DEAM is also the only external judgement in the entire project: each track was rated by independent listeners, and those ratings directly shape how the music is visualised by the AI. That adds a layer of cultural bias to the inner workings of machine and human design.


**Stage 3 — CLAP (378 tracks, zero-shot labels).**
CLAP (Wu et al. 2023) embeds audio and text into one space, so a track can be scored against a written description instead of a formula.

```bash
python -m server.clap_labels --heads color_temperature complexity motion_speed style
python -m server.train --balance-classes
```

Three out of four heads worked. Colour temperature went from a 0.083 standard deviation to **0.339**. A track called "Dark Matter" scores 0.025 while a disco track scores 0.828. Complexity exposed a bug where the formula was saturated, making it hard to differentiate between complex and very complex.
The three that worked ask CLAP to place a track on an axis (cold against warm, sparse against dense, slow against fast), with two opposed anchors that normalise each other. Style asks it to choose among seven descriptions of visual forms nobody has ever captioned audio with, so the similarities come back nearly identical and argmax returns the most generic prompt rather than the best fit. This is why the fourth didn't work.

### Experimentation and iterations

Now that the visuals worked and the mechanism behind them was coherent, the next question was whether the model was actually improving. The first thing that turned up was that `random_split` had no seed, which meant no two checkpoints had ever been comparable. The split is now seeded (`SPLIT_SEED`) and every change is judged per head:

```bash
python -m server.evaluate trained_models/visgen_cnn.pt trained_models/visgen_new.pt
```
`server/evaluate.py`

- Number outputs: MAE, R² and prediction spread.
- Class outputs: accuracy and macro-F1.

Macro-F1 carries the weight: mood is 56% neutral, so a model that answers neutral every time scores well on accuracy and is worthless.

The first thing it caught was also the most important: the averaged training loss had fallen from 0.228 to 0.131. At that point a head had been removed, so the mean was being taken across eight heads instead of nine.

Three experiments then failed:

Class weighting. The model still labelled 56% of tracks "neutral", which did not produce more varied output, but it unexpectedly made the model much better at telling a happy track from a sad one — which was not what the change was for.

SpecAugment. Blanking out random slices of each track during training, a standard trick for stopping a model from simply memorising its examples, made things worse almost everywhere here: instead of helping the model generalise, the evidence it hid turned out to be crucial.

Removing the mood output. Deleting it seemed like a good idea, as the visualizer never shows it, but seven of the eight remaining outputs immediately got worse — because learning to put a name to a track's mood had been teaching the network to weigh emotion into the visualised output.

---

## Project structure

```
index.html                 Web app markup
static/css/index-style.css Web app styles
static/js/                 Visualizer: formulas, rendering, audio, export, app shell

index.py                   Flask server: serves the app and POST /api/analyze
server/audio_analyzer.py   librosa features, Foote structural segmentation, mood scalars
server/dataset_builder.py  Labelling, spectrogram building, masked Dataset classes
server/model.py            VisgenCNN + VisgenEmbeddingNet, shared multi-task heads
server/train.py            Training loop, uncertainty-weighted loss, SpecAugment
server/evaluate.py         Per-head validation metrics on the seeded split
server/deam.py             DEAM corpus ingestion (human valence/arousal)
server/clap_labels.py      Zero-shot labels for the heads no corpus annotates
server/clap_embeddings.py  CLAP audio embeddings, disk-cached
tests/test_masking.py      Guards the per-head label masking

requirements.txt           Python dependencies
Dockerfile                 Container image (Hugging Face Spaces, or any Docker host)
dataset/                   metadata.csv, plus the stage1-bootstrap and stage2-deam snapshots
trained_models/            visgen_cnn.pt is the served checkpoint
music-test/                One track, so the app can be tried without supplying your own
docs/                      Visgen-documentation.pdf - build process, experiments, evaluation
```

