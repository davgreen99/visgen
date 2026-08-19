import { $, $$, clamp, PALETTE_HEX, aiPaletteHex } from './utils.js';
import { FORMULAS, formulaDefaults } from './formulas.js';
import { AudioEngine, analyzeAudio } from './audio.js';
import { Visualizer } from './visualizer.js';
import { exportVideo, exportSupported } from './exporter.js';
import { apiUrl } from './config.js';

const WAVEFORM_BARS = 72;

class App {
    constructor() {
        this.audio = new AudioEngine();
        this.visualizer = null;
        this.analysis = null;
        this.state = { audioReactive: true };
        this.layers = [{ style: "harmonograph", params: formulaDefaults("harmonograph"), colors: PALETTE_HEX.warm.slice(), autoPalette: true, offset: { x: 0, y: 0 } }];
        this.active = 0;
        this.maxLayers = 4;
        this.layersEnabled = false;
        this._progressRAF = null;
        this.trackName = null;
        this.creations = this._loadCreations();
        this._init();
    }

    _init() {
        this.fileInput = Object.assign(document.createElement("input"), { type: "file", accept: ".mp3,.wav,.flac,.aac,.ogg,.m4a", style: "display:none" });
        document.body.appendChild(this.fileInput);
        this.fileInput.addEventListener("change", () => { if (this.fileInput.files.length) this._handleFile(this.fileInput.files[0]); });

        $(".waveform").append(...Array.from({ length: WAVEFORM_BARS },
            () => Object.assign(document.createElement("div"), { className: "waveform-bar" })));

        const zone = $(".upload-zone");
        zone.addEventListener("click", () => this.fileInput.click());
        $(".upload-zone .btn").addEventListener("click", e => { e.stopPropagation(); this.fileInput.click(); });
        zone.addEventListener("dragover", e => { e.preventDefault(); zone.style.borderColor = "var(--color-accent)"; zone.style.background = "var(--color-accent-subtle)"; });
        zone.addEventListener("dragleave", () => { zone.style.borderColor = ""; zone.style.background = ""; });
        zone.addEventListener("drop", e => { e.preventDefault(); zone.style.borderColor = ""; zone.style.background = ""; if (e.dataTransfer.files.length) this._handleFile(e.dataTransfer.files[0]); });

        // Player seek
        const bar = $(".progress-bar");
        let dragging = false;
        const seek = e => {
            if (!this.audio.audioElement) return;
            const r = bar.getBoundingClientRect();
            const x = e.clientX ?? e.touches?.[0]?.clientX ?? 0;
            this.audio.audioElement.currentTime = clamp((x - r.left) / r.width, 0, 1) * this.audio.audioElement.duration;
            this._renderProgress();
        };
        bar.addEventListener("pointerdown", e => { if (!this.audio.audioElement) return; dragging = true; bar.classList.add("progress-bar--dragging"); bar.setPointerCapture(e.pointerId); seek(e); });
        bar.addEventListener("pointermove", e => { if (dragging) seek(e); });
        bar.addEventListener("pointerup", e => { if (dragging) { dragging = false; bar.classList.remove("progress-bar--dragging"); } });

        $(".play-btn").addEventListener("click", () => {
            if (!this.audio.audioElement) return;
            this.audio.toggle();
            $(".play-btn").classList.toggle("play-btn--playing", this.audio.playing);
            if (this.audio.playing) this._startProgressLoop();
            this._setImmersive(this.audio.playing);
        });

        // Formula buttons
        $$('[data-control="style"] .option-btn').forEach(btn => btn.addEventListener("click", () => {
            this._selectFormula(btn.dataset.value);
        }));
        // Palette presets
        $$('[data-control="palette"] .option-btn').forEach(btn => btn.addEventListener("click", () => {
            const v = btn.dataset.value, layer = this.layers[this.active];
            layer.autoPalette = v === "auto";
            layer.colors = v === "auto"
                ? (this.analysis ? aiPaletteHex(this.analysis) : PALETTE_HEX.warm.slice())
                : PALETTE_HEX[v].slice();
            this._renderSwatches();
            if (this.visualizer) this.visualizer.setLayerPalette(this.active, layer.colors);
        }));

        // "+ Add Layer" button
        $(".controls-add-layer")?.addEventListener("click", () => this._addLayer());

        // "Combine Layers" toggle
        const layersToggle = document.getElementById("layers-toggle");
        if (layersToggle) layersToggle.addEventListener("change", () => {
            this.layersEnabled = layersToggle.checked;
            if (!this.layersEnabled && this.layers.length > 1) {
                this.layers = [this.layers[this.active]];
                this.active = 0;
                if (this.visualizer) this.visualizer.setLayers(this.layers);
            }
            this._updateLayersUI();
        });

        // Layer chips, swatches, sliders
        this._syncStyleButtons();
        this._renderParamSliders(this.layers[this.active].style);
        this._renderSwatches();
        this._updateLayersUI();

        // Audio Reactive toggle
        const reactiveEl = document.getElementById("reactive");
        if (reactiveEl) reactiveEl.addEventListener("change", () => {
            this.state.audioReactive = reactiveEl.checked;
            if (this.visualizer) this.visualizer.setReactive(reactiveEl.checked);
        });

        // Actions
        $(".btn--full").addEventListener("click", () => this._generate());
        $('.icon-btn[title="Fullscreen"]')?.addEventListener("click", () => {
            document.fullscreenElement ? document.exitFullscreen() : $(".canvas-panel-display").requestFullscreen().catch(() => {});
        });
        $('.icon-btn[title="Screenshot"]')?.addEventListener("click", () => {
            if (!this.visualizer) return;
            const a = Object.assign(document.createElement("a"), { href: this.visualizer.snapshot(), download: "visgen-" + Date.now() + ".png" });
            a.click();
        });
        $('.icon-btn[title="Download"]')?.addEventListener("click", () => this._confirmDownloadVideo());
        $('.icon-btn[title="Share"]')?.addEventListener("click", () => this._shareLink());

        this._bindLayerDrag();
        this._renderGallery();
        this._applySharedState();
    }

    // Layer drag - moves the selected layer, not what is under the pointer
    _bindLayerDrag() {
        const surface = $(".canvas-panel-display");
        if (!surface) return;

        let dragging = false, lastX = 0, lastY = 0, perPx = 0;

        const draggable = () => {
            const layer = this.layers[this.active];
            return this.visualizer && layer && !FORMULAS[layer.style]?.fill ? layer : null;
        };

        surface.addEventListener("pointerdown", e => {
            if (e.button !== 0 || !draggable()) return;
            dragging = true;
            lastX = e.clientX; lastY = e.clientY;
            perPx = this.visualizer.worldPerPixel(surface.clientHeight);
            surface.setPointerCapture?.(e.pointerId);
            surface.classList.add("is-dragging-layer");
        });

        surface.addEventListener("pointermove", e => {
            const layer = dragging && draggable();
            if (!layer) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            const next = this.visualizer.setLayerOffset(this.active,
                layer.offset.x + dx * perPx, layer.offset.y - dy * perPx);
            if (next) layer.offset = { x: next.x, y: next.y };
        });

        const end = e => {
            if (!dragging) return;
            dragging = false;
            surface.releasePointerCapture?.(e.pointerId);
            surface.classList.remove("is-dragging-layer");
        };
        surface.addEventListener("pointerup", end);
        surface.addEventListener("pointercancel", end);

        // Double-click recentres
        surface.addEventListener("dblclick", () => {
            const layer = draggable();
            if (!layer) return;
            layer.offset = { x: 0, y: 0 };
            this.visualizer.setLayerOffset(this.active, 0, 0);
        });
    }

    // Grab cursor state
    _syncDragCursor() {
        const surface = $(".canvas-panel-display");
        const layer = this.layers[this.active];
        surface?.classList.toggle("is-layer-locked", !!FORMULAS[layer?.style]?.fill);
    }

    // Share link - layers packed into a URL-safe token
    _shareLink() {
        const state = {
            v: 1,
            reactive: this.state.audioReactive,
            layers: this.layers.map(l => ({ style: l.style, params: l.params, colors: l.colors, offset: l.offset })),
        };
        const json = JSON.stringify(state);
        const b64 = btoa(String.fromCharCode(...new TextEncoder().encode(json)))
            .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
        const url = `${location.origin}${location.pathname}#s=${b64}`;
        const btn = $('.icon-btn[title="Share"]');
        navigator.clipboard?.writeText(url).then(() => {
            if (btn) { btn.style.color = "var(--color-accent)"; setTimeout(() => btn.style.color = "", 800); }
            alert("Share link copied to clipboard.\nAnyone opening it gets these formulas, parameters and colours — they add their own track.");
        }).catch(() => prompt("Copy this share link:", url));
    }

    // Apply layers/params from the URL hash
    _applySharedState() {
        const m = location.hash.match(/[#&]s=([A-Za-z0-9_-]+)/);
        if (!m) return;
        try {
            const bytes = Uint8Array.from(atob(m[1].replace(/-/g, "+").replace(/_/g, "/")), c => c.charCodeAt(0));
            const state = JSON.parse(new TextDecoder().decode(bytes));
            const layers = (state.layers || []).filter(l => FORMULAS[l.style]);
            if (!layers.length) return;
            this.layers = layers.slice(0, this.maxLayers).map(l => ({
                style: l.style,
                params: { ...formulaDefaults(l.style), ...(l.params || {}) },
                colors: Array.isArray(l.colors) && l.colors.length ? l.colors.slice(0, 4) : PALETTE_HEX.warm.slice(),
                autoPalette: false,
                offset: { x: Number(l.offset?.x) || 0, y: Number(l.offset?.y) || 0 },
            }));
            this.active = 0;
            this.state.audioReactive = state.reactive !== false;
            this.layersEnabled = this.layers.length > 1;
            const toggle = document.getElementById("layers-toggle");
            if (toggle) toggle.checked = this.layersEnabled;
            const reactiveEl = document.getElementById("reactive");
            if (reactiveEl) reactiveEl.checked = this.state.audioReactive;
            this._fromShare = true;
            this._syncStyleButtons();
            this._renderParamSliders(this.layers[0].style);
            this._renderSwatches();
            this._updateLayersUI();
        } catch (e) { console.warn("Ignoring invalid share link:", e); }
    }

    // Formula defaults merged with the per-track seed
    _seededParams(id) {
        const seed = (this.analysis && FORMULAS[id].seed) ? FORMULAS[id].seed(this.analysis) : {};
        return { ...formulaDefaults(id), ...seed };
    }

    // Switch formula and reseed for the current track
    _selectFormula(id) {
        const layer = this.layers[this.active];
        layer.style = id;
        layer.params = this._seededParams(id);
        this._syncStyleButtons();
        this._renderParamSliders(id);
        this._renderLayerChips();
        this._syncDragCursor();
        if (this.visualizer) this.visualizer.setLayerMode(this.active, id, layer.params);
    }

    // Highlight the active layer's formula button
    _syncStyleButtons() {
        const style = this.layers[this.active].style;
        $$('[data-control="style"] .option-btn').forEach(b =>
            b.classList.toggle("option-btn--active", b.dataset.value === style));
    }

    // Layer chips (select / remove) and "+ Add"
    _renderLayerChips() {
        const c = $(".controls-layers"); if (!c) return;
        c.innerHTML = "";
        this.layers.forEach((l, i) => {
            const chip = document.createElement("button");
            chip.className = "option-btn" + (i === this.active ? " option-btn--active" : "");
            chip.textContent = FORMULAS[l.style].label;
            chip.addEventListener("click", () => this._selectLayer(i));
            if (this.layers.length > 1) {
                const x = document.createElement("span");
                x.textContent = " ×"; x.style.opacity = "0.6";
                x.addEventListener("click", e => { e.stopPropagation(); this._removeLayer(i); });
                chip.appendChild(x);
            }
            c.appendChild(chip);
        });
        const add = $(".controls-add-layer");
        if (add) add.style.display = this.layers.length < this.maxLayers ? "" : "none";
    }

    // Show/hide the layer chips
    _updateLayersUI() {
        const group = $('[data-control="layers"]');
        if (group) group.style.display = this.layersEnabled ? "" : "none";
        $$(".controls-hint").forEach(h => h.style.display = this.layersEnabled ? "" : "none");
        if (this.layersEnabled) this._renderLayerChips();
        this._syncDragCursor();
    }

    _selectLayer(i) {
        this.active = i;
        this._syncStyleButtons();
        this._renderParamSliders(this.layers[i].style);
        this._renderSwatches();
        this._renderLayerChips();
        this._syncDragCursor();
    }

    // Immersive mode - visual fills the page background
    _setImmersive(on) {
        const go = on && !!this.visualizer;
        document.body.classList.toggle("is-immersive", go);
        if (!this.visualizer) return;
        if (go) this.visualizer.enterImmersive($("#visual-bg"), $(".canvas-panel-display"));
        else this.visualizer.exitImmersive($(".canvas-panel-display"));
    }

    // Colour pickers for the active layer
    _renderSwatches() {
        const c = $(".controls-swatches"); if (!c) return;
        const layer = this.layers[this.active];
        c.innerHTML = "";
        layer.colors.forEach((hex, i) => {
            const inp = Object.assign(document.createElement("input"), { type: "color", value: hex });
            inp.addEventListener("input", () => {
                layer.colors[i] = inp.value; layer.autoPalette = false;
                if (this.visualizer) this.visualizer.setLayerPalette(this.active, layer.colors);
            });
            c.appendChild(inp);
        });
    }

    _addLayer() {
        if (!this.layersEnabled || this.layers.length >= this.maxLayers) return;
        const used = this.layers.map(l => l.style);
        const order = ["harmonograph", "dejong", "chladni", "waveform", "lorenz", "thomas", "aizawa"];
        const style = order.find(s => !used.includes(s)) || "dejong";
        const presets = [PALETTE_HEX.cool, PALETTE_HEX.mono, PALETTE_HEX.warm];
        const def = { style, params: this._seededParams(style), colors: presets[(this.layers.length - 1) % presets.length].slice(), autoPalette: false, offset: { x: 0, y: 0 } };
        this.layers.push(def);
        this.active = this.layers.length - 1;
        if (this.visualizer) this.visualizer.addLayer(def);
        this._syncStyleButtons();
        this._renderParamSliders(style);
        this._renderSwatches();
        this._renderLayerChips();
    }

    _removeLayer(i) {
        if (this.layers.length <= 1) return;
        this.layers.splice(i, 1);
        if (this.visualizer) this.visualizer.removeLayer(i);
        if (this.active >= this.layers.length) this.active = this.layers.length - 1;
        else if (this.active > i) this.active--;
        this._selectLayer(this.active);
    }

    // One slider per formula parameter
    _renderParamSliders(id) {
        const container = $('[data-control="params"]'); if (!container) return;
        const params = FORMULAS[id].params, layer = this.layers[this.active];
        container.innerHTML = "";
        for (const key in params) {
            const def = params[key];
            const val = layer.params[key] ?? def.value;
            const group = document.createElement("div");
            group.className = "controls-group";
            group.innerHTML =
                `<label class="controls-label">${def.label}` +
                `<span class="controls-value">${this._fmtParam(val, def.step)}</span></label>` +
                `<input type="range" class="controls-slider" min="${def.min}" max="${def.max}" step="${def.step}" value="${val}">`;
            const slider = group.querySelector(".controls-slider");
            const valEl = group.querySelector(".controls-value");
            slider.addEventListener("input", () => {
                const v = parseFloat(slider.value);
                this.layers[this.active].params[key] = v;
                valEl.textContent = this._fmtParam(v, def.step);
                if (this.visualizer) this.visualizer.setLayerParam(this.active, key, v);
            });
            container.appendChild(group);
        }
    }

    _fmtParam(v, step) { return step >= 1 ? String(Math.round(v)) : Number(v).toFixed(2); }

    async _handleFile(file) {
        const ext = file.name.split(".").pop().toLowerCase();
        if (!["mp3","wav","flac","aac","ogg","m4a"].includes(ext)) { alert("Unsupported file type."); return; }

        $(".track-card-name").textContent = file.name;
        this.trackName = file.name;
        this.trackFile = file;
        $(".track-card-duration").textContent = "analyzing...";
        $(".upload-zone").style.borderColor = "var(--color-accent)";
        $(".upload-zone").classList.add("upload-zone--loading");
        $(".upload-zone-title").textContent = "Analyzing...";

        try {
            this.analysis = await analyzeAudio(file);
            this._updateChips();
            this._updateWaveform();
            $(".track-card-duration").textContent = this._fmt(this.analysis.duration);
            $(".upload-zone").classList.remove("upload-zone--loading");
            $(".upload-zone-title").textContent = "Track loaded";
            $(".upload-zone-sub").textContent = "Drop another to replace";

            if (this._fromShare) {
                this._fromShare = false;
                if (this.visualizer) { this.visualizer.setAnalysis(this.analysis); this.visualizer.setLayers(this.layers); }
            } else if (this.analysis.cnn) {
                const predicted = this.analysis.cnn.style;
                const formula = FORMULAS[predicted] ? predicted : "harmonograph";
                this.layers.forEach(l => {
                    l.params = this._seededParams(l.style);
                    if (l.autoPalette) l.colors = aiPaletteHex(this.analysis);
                });
                this.layers[this.active].style = formula;
                this.layers[this.active].params = this._seededParams(formula);
                this._syncStyleButtons();
                this._renderParamSliders(formula);
                this._renderSwatches();
                this._renderLayerChips();
                if (this.visualizer) { this.visualizer.setAnalysis(this.analysis); this.visualizer.setLayers(this.layers); }
            }

            await this.audio.loadFile(file);
            $(".play-btn").disabled = false;
            $(".play-btn").classList.remove("play-btn--playing");
            $(".progress-bar-fill").style.width = "0%";
            $(".progress-bar-head").style.left = "0%";
            $(".track-card-current").textContent = "0:00";
            this.audio.audioElement.addEventListener("ended", () => { $(".play-btn").classList.remove("play-btn--playing"); this.audio.playing = false; this._renderProgress(); this._setImmersive(false); });
        } catch (err) {
            $(".track-card-duration").textContent = "error";
            $(".upload-zone").classList.remove("upload-zone--loading");
            $(".upload-zone-title").textContent = "Analysis failed";
            $(".upload-zone").style.borderColor = "";
            console.error("Analysis error:", err);
            alert("Failed to analyze: " + err.message);
        }
    }

    _updateChips() {
        if (!this.analysis) return;
        const a = this.analysis, c = a.cnn || {};
        const data = [`BPM: ${a.bpm||"—"}`, `Key: ${a.key||"—"}`, `Mood: ${c.mood||a.mood||"—"}`,
            `Energy: ${c.energy != null ? (c.energy*100).toFixed(0)+"%" : "—"}`,
            `Timbre: ${a.timbre_brightness != null ? (a.timbre_brightness > 0 ? "Bright" : "Dark") : "—"}`];
        $$(".chip").forEach((chip, i) => { chip.textContent = data[i] || ""; chip.style.borderColor = "var(--color-accent-border)"; chip.style.color = "var(--color-text-secondary)"; });
    }

    // Shape the bar row to the track's energy timeline
    _updateWaveform() {
        const tl = this.analysis?.energy_timeline; if (!tl) return;
        const bars = $$(".waveform-bar"), step = Math.max(1, Math.floor(tl.length / bars.length));
        const max = Math.max(...tl) || 1;
        $(".waveform").classList.add("waveform--loaded");
        bars.forEach((bar, i) => {
            const h = Math.max(5, (tl[Math.min(i * step, tl.length - 1)] / max) * 100);
            bar.style.height = h + "%";
            bar.style.opacity = bar.dataset.baseOpacity = 0.3 + (h / 100) * 0.4;
        });
    }

    async _generate() {
        if (!this.analysis) { alert("Upload a track first."); return; }
        const empty = $(".canvas-empty"); if (empty) empty.style.display = "none";
        if (!this.visualizer) this.visualizer = new Visualizer($(".canvas-panel-display"));
        this.visualizer.setReactive(this.state.audioReactive);
        this.visualizer.setAnalysis(this.analysis);
        this.visualizer.setLayers(this.layers);
        this.visualizer.start();
        this.audio.play();
        $(".play-btn").classList.add("play-btn--playing");
        this._setImmersive(true);
        this._startProgressLoop();
        this._feedLoop();
    }

    // Feed the visualizer live spectrum + playback position
    _feedLoop() {
        if (!this.visualizer?.running) return;
        this.visualizer.feedAudio(this.audio.getBands(), this.audio.audioElement?.currentTime ?? 0);
        requestAnimationFrame(() => this._feedLoop());
    }

    _startProgressLoop() {
        if (this._progressRAF) return;
        const tick = () => { this._renderProgress(); this._progressRAF = this.audio.playing ? requestAnimationFrame(tick) : null; };
        this._progressRAF = requestAnimationFrame(tick);
    }

    _renderProgress() {
        const el = this.audio.audioElement; if (!el?.duration) return;
        const pct = (el.currentTime / el.duration) * 100;
        $(".progress-bar-fill").style.width = pct + "%";
        $(".progress-bar-head").style.left = pct + "%";
        $(".track-card-current").textContent = this._fmt(el.currentTime);
        const active = Math.floor((pct / 100) * $$(".waveform-bar").length);
        $$(".waveform-bar").forEach((b, i) => { b.style.opacity = i <= active ? "1" : b.dataset.baseOpacity || "0.5"; });
    }

    _fmt(s) { return s ? Math.floor(s/60) + ":" + String(Math.floor(s%60)).padStart(2,"0") : "0:00"; }

    // Download confirmation
    _confirmDownloadVideo() {
        if (!this.visualizer || !this.audio.audioElement) { alert("Upload a track and Generate a visual first."); return; }
        if (this._rendering) return;
        const modal = $("#download-modal"); if (!modal) return;
        const cancelBtn = modal.querySelector('[data-modal="cancel"]');
        const confirmBtn = modal.querySelector('[data-modal="confirm"]');
        const done = ok => {
            modal.hidden = true;
            cancelBtn.removeEventListener("click", onCancel);
            confirmBtn.removeEventListener("click", onConfirm);
            if (ok) this._downloadVideo();
        };
        const onCancel = () => done(false), onConfirm = () => done(true);
        cancelBtn.addEventListener("click", onCancel);
        confirmBtn.addEventListener("click", onConfirm);
        modal.hidden = false;
    }

    async _downloadVideo() {
        if (this._rendering) return;
        if (!exportSupported() || !this.trackFile) {
            const missing = [
                typeof VideoEncoder === "undefined" && "WebCodecs video encoding (use Chrome or Edge)",
                typeof Mp4Muxer === "undefined" && "the mp4-muxer script (hard-refresh the page: Ctrl+F5)",
                !this.trackFile && "the uploaded track (re-upload it)",
            ].filter(Boolean).join(", ");
            console.warn("Offline render unavailable — missing:", missing);
            alert(`Can't render the video — missing: ${missing}.`);
            return;
        }

        this._rendering = true;
        this.audio.pause();
        $(".play-btn").classList.remove("play-btn--playing");

        const banner = $("#render-banner"), text = banner.querySelector(".render-banner-text");
        const cancelBtn = $("#render-cancel");
        let cancelled = false;
        const onCancel = () => { cancelled = true; };
        cancelBtn.addEventListener("click", onCancel);
        banner.hidden = false;

        const prevTitle = document.title;
        let wakeLock = null;
        const acquireLock = () => navigator.wakeLock?.request("screen").then(l => { wakeLock = l; }).catch(() => {});
        const onVis = () => { if (document.visibilityState === "visible") acquireLock(); };
        acquireLock();
        document.addEventListener("visibilitychange", onVis);

        let failed = null;
        try {
            const blob = await exportVideo({
                visualizer: this.visualizer,
                file: this.trackFile,
                onProgress: msg => { text.textContent = msg; document.title = `⏳ ${msg} — Visgen`; },
                isCancelled: () => cancelled,
            });
            if (blob) {
                const url = URL.createObjectURL(blob);
                Object.assign(document.createElement("a"), { href: url, download: `visgen-${Date.now()}.mp4` }).click();
                setTimeout(() => URL.revokeObjectURL(url), 10000);
                this._saveCreation();
            }
        } catch (err) {
            failed = err;
        } finally {
            document.title = prevTitle;
            document.removeEventListener("visibilitychange", onVis);
            wakeLock?.release().catch(() => {});
            banner.hidden = true;
            cancelBtn.removeEventListener("click", onCancel);
            this._rendering = false;
        }
        if (failed) {
            console.error("Offline render failed:", failed);
            alert(`Rendering failed: ${failed.message}`);
        }
    }

    /* Gallery of saved creations */

    _loadCreations() {
        try { return JSON.parse(localStorage.getItem("visgen-creations")) || []; }
        catch { return []; }
    }
    _persistCreations() {
        try { localStorage.setItem("visgen-creations", JSON.stringify(this.creations)); }
        catch (e) { console.warn("Gallery not saved (storage full?)", e); }
    }

    // Snapshot formulas, params, colours and track into the gallery
    _saveCreation() {
        if (!this.visualizer || !this.analysis) return;
        const a = this.analysis, c = a.cnn || {};
        this.creations.unshift({
            id: "c" + Date.now(),
            track: this.trackName || "Untitled track",
            date: Date.now(),
            thumb: this.visualizer.thumbnail(),
            chips: { bpm: a.bpm, key: a.key, mood: c.mood || a.mood },
            layers: this.layers.map(l => ({ style: l.style, params: { ...l.params }, colors: l.colors.slice(), offset: { ...l.offset } })),
        });
        if (this.creations.length > 12) this.creations.length = 12;
        this._persistCreations();
        this._renderGallery();
        const btn = $('.icon-btn[title="Download"]');
        if (btn) { btn.style.color = "var(--color-accent)"; setTimeout(() => btn.style.color = "", 600); }
    }

    // Rebuild the gallery grid
    _renderGallery() {
        const grid = $(".gallery"); if (!grid) return;
        grid.querySelectorAll(".gallery-item--saved").forEach(el => el.remove());
        grid.querySelectorAll(".gallery-item:not(.gallery-item--saved)")
            .forEach(d => d.style.display = this.creations.length ? "none" : "");
        const frag = document.createDocumentFragment();
        this.creations.forEach(cr => frag.appendChild(this._galleryCard(cr)));
        grid.insertBefore(frag, grid.firstChild);
    }

    _galleryCard(cr) {
        const summary = cr.layers.map(l => FORMULAS[l.style]?.label || l.style).join(" · ");
        const item = document.createElement("div");
        item.className = "gallery-item gallery-item--saved";
        item.dataset.id = cr.id;

        const ph = document.createElement("div");
        ph.className = "gallery-placeholder";
        if (cr.thumb) { ph.style.backgroundImage = `url(${cr.thumb})`; ph.style.backgroundSize = "cover"; ph.style.backgroundPosition = "center"; }
        item.appendChild(ph);

        const overlay = document.createElement("div");
        overlay.className = "gallery-overlay";
        overlay.innerHTML = `<span class="gallery-track">${this._esc(cr.track)}</span><span class="gallery-style">${this._esc(summary)}</span>`;
        item.appendChild(overlay);

        const details = document.createElement("div");
        details.className = "gallery-details";
        details.innerHTML = this._creationDetailsHTML(cr);
        item.appendChild(details);

        const add = document.createElement("button");
        add.className = "gallery-add";
        add.textContent = "+ Add to visualizer";
        add.addEventListener("click", e => { e.stopPropagation(); this._loadCreation(cr.id); });
        item.appendChild(add);

        const del = document.createElement("button");
        del.className = "gallery-remove"; del.title = "Delete"; del.textContent = "×";
        del.addEventListener("click", e => { e.stopPropagation(); this._removeCreation(cr.id); });
        item.appendChild(del);

        item.addEventListener("click", () => item.classList.toggle("is-open"));
        return item;
    }

    _creationDetailsHTML(cr) {
        const ch = cr.chips || {};
        const meta = [ch.bpm ? `BPM ${ch.bpm}` : null, ch.key ? `Key ${ch.key}` : null, ch.mood || null].filter(Boolean).join(" · ");
        let html = `<h4>${this._esc(cr.track)}</h4>`;
        if (meta) html += `<p class="gallery-meta">${this._esc(meta)}</p>`;
        cr.layers.forEach(l => {
            const f = FORMULAS[l.style], params = f?.params || {};
            html += `<div class="gallery-layer"><strong>${this._esc(f?.label || l.style)}</strong><ul>`;
            for (const k in params) {
                if (l.params[k] == null) continue;
                html += `<li>${this._esc(params[k].label)}: ${this._fmtParam(l.params[k], params[k].step)}</li>`;
            }
            html += `</ul></div>`;
        });
        return html;
    }

    // Restore a saved creation into the controls
    _loadCreation(id) {
        const cr = this.creations.find(c => c.id === id); if (!cr) return;
        this.layers = cr.layers.map(l => ({
            style: l.style,
            params: { ...formulaDefaults(l.style), ...l.params },
            colors: (l.colors && l.colors.slice()) || PALETTE_HEX.warm.slice(),
            autoPalette: false,
            offset: { x: Number(l.offset?.x) || 0, y: Number(l.offset?.y) || 0 },
        }));
        this.active = 0;
        this.layersEnabled = this.layers.length > 1;
        const lt = document.getElementById("layers-toggle"); if (lt) lt.checked = this.layersEnabled;
        this._syncStyleButtons();
        this._renderParamSliders(this.layers[0].style);
        this._renderSwatches();
        this._updateLayersUI();
        if (this.visualizer) this.visualizer.setLayers(this.layers);
        document.getElementById("app")?.scrollIntoView({ behavior: "smooth" });
    }

    _removeCreation(id) {
        this.creations = this.creations.filter(c => c.id !== id);
        this._persistCreations();
        this._renderGallery();
    }

    _esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
}

// Nav auto-hide away from the top of the page
function initNavAutoHide() {
    const nav = $(".nav"); if (!nav) return;
    const NAV_HEIGHT = 120;
    addEventListener("scroll", () => {
        const y = Math.max(0, window.scrollY);
        nav.classList.toggle("nav--collapsed", y > NAV_HEIGHT);
    }, { passive: true });
}

document.addEventListener("DOMContentLoaded", () => {
    window.visgenApp = new App();
    initNavAutoHide();
});
