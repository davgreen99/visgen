/* Track analysis via /api/analyze - no client-side fallback */

import { BACKEND_URL, apiUrl } from './config.js';

async function analyzeAudio(file) {
    const form = new FormData();
    form.append("file", file);

    let res;
    try {
        res = await fetch(apiUrl("/api/analyze"), { method: "POST", body: form });
    } catch (e) {
        throw new Error(BACKEND_URL
            ? `Can't reach the analysis backend at ${BACKEND_URL}. It may be asleep — wait a moment and try again.`
            : "Can't reach the analysis server. Start it with:  python index.py");
    }
    if (!res.ok) {
        let msg = `Analysis failed (${res.status})`;
        try { const j = await res.json(); if (j.error) msg = j.error; } catch {}
        throw new Error(msg);
    }

    const data = await res.json();
    if (!data.cnn) {
        throw new Error("The AI model isn't loaded. Train it (python -m server.train) so a checkpoint exists at trained_models/visgen_cnn.pt, then restart the server.");
    }
    data.cnn.motion = data.cnn.motion ?? data.cnn.motion_speed;
    return data;
}

/* AudioEngine - playback + live frequency bands */

class AudioEngine {
    constructor() {
        this.ctx = null; this.analyser = null; this.source = null;
        this.audioElement = null; this.playing = false;
        this.freqData = null;
    }

    async loadFile(file) {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.ctx.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;
            this.analyser.connect(this.ctx.destination);
        }
        this.stop();
        const url = URL.createObjectURL(file);
        this.audioElement = new Audio(url);
        this.audioElement.crossOrigin = "anonymous";
        await new Promise((res, rej) => {
            this.audioElement.addEventListener("canplaythrough", res, { once: true });
            this.audioElement.addEventListener("error", rej, { once: true });
            this.audioElement.load();
        });
        this.source = this.ctx.createMediaElementSource(this.audioElement);
        this.source.connect(this.analyser);
        this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
    }

    play()   { if (!this.audioElement) return; if (this.ctx.state === "suspended") this.ctx.resume(); this.audioElement.play(); this.playing = true; }
    pause()  { if (this.audioElement) this.audioElement.pause(); this.playing = false; }
    stop()   { if (this.audioElement) { this.audioElement.pause(); this.audioElement.currentTime = 0; } this.playing = false; }
    toggle() { this.playing ? this.pause() : this.play(); }

    getBands() {
        if (!this.analyser || !this.freqData) return { bass: 0, mid: 0, high: 0, overall: 0 };
        this.analyser.getByteFrequencyData(this.freqData);
        const len = this.freqData.length;
        const bEnd = Math.floor(len * 0.1), mEnd = Math.floor(len * 0.5);
        let bass = 0, mid = 0, high = 0;
        for (let i = 0; i < bEnd; i++) bass += this.freqData[i];
        for (let i = bEnd; i < mEnd; i++) mid += this.freqData[i];
        for (let i = mEnd; i < len; i++) high += this.freqData[i];
        bass /= bEnd * 255; mid /= (mEnd - bEnd) * 255; high /= (len - mEnd) * 255;
        return { bass, mid, high, overall: (bass + mid + high) / 3 };
    }
}

export { analyzeAudio, AudioEngine };
