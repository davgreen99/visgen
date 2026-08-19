import { clamp, lerp, PALETTE_HEX } from './utils.js';
import { FORMULAS, formulaDefaults, REACT, compileFormula, CHLADNI_VERT, CHLADNI_FRAG } from './formulas.js';

/* Visualizer - independent layers, one formula each, eased toward new targets */

class Visualizer {
    constructor(container) {
        this.container = container;
        this.analysis = null;
        this.clock = new THREE.Clock();
        this.layers = [];
        this.objects = {};
        this.running = false;
        this._reactive = true;
        this._morphRate = 3;
        this._currentBands = { bass: 0, mid: 0, high: 0, overall: 0 };
        this._trackTime = 0; this._beats = []; this._beatIdx = 0; this._beatClock = 0;
        this._segments = []; this._trackVec = null; this._segIndex = -1; this._beat = 0;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        const w = container.clientWidth, h = container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
        this.camera.position.set(0, 0, 5.4);
        this.camera.lookAt(0, 0, 0);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        this.composer = null;
        if (typeof THREE.EffectComposer !== "undefined") {
            this.composer = new THREE.EffectComposer(this.renderer);
            this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));
            this.bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(w, h), 0.8, 0.4, 0.85);
            this.composer.addPass(this.bloomPass);
        }
        window.addEventListener("resize", () => this.resize());
    }

    /* Public API (driven by App) */

    setReactive(v) { this._reactive = v; }

    setAnalysis(d) {
        this.analysis = d;
        this._beats = (d?.beat_times || []).map(Number);
        this._segments = d?.structure?.segments || [];
        this._trackVec = d?.structure?.track || null;
        this._beatIdx = 0; this._beatClock = 0; this._segIndex = -1; this._cnnVec = null;
        this.layers.forEach(l => { l.trackSeed = null; });
        this._deriveMorphRate();
        this.layers.forEach((l, i) => this._deriveMotion(l, i));
    }

    // `time` is playback position in the track, not wall time
    feedAudio(b, time) {
        this._currentBands = b;
        if (typeof time === "number" && Number.isFinite(time)) this._trackTime = time;
    }

    // Per-layer palette
    setLayerPalette(i, hexColors) {
        const l = this.layers[i]; if (!l) return;
        l.palette = hexColors.map(h => new THREE.Color(h));
        l.needsRebuild = true;
    }

    // Re-fit renderer and camera to the current host element
    resize() {
        const host = this.renderer.domElement.parentElement || this.container;
        const w = host.clientWidth, h = host.clientHeight;
        if (!w || !h) return;
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        if (this.composer) this.composer.setSize(w, h);
    }

    // Enter immersive - canvas becomes the page background, box shows a mirror
    enterImmersive(bgHost, boxHost) {
        if (!bgHost || !boxHost) return;
        bgHost.appendChild(this.renderer.domElement);
        if (!this._mirror) {
            this._mirror = document.createElement("canvas");
            this._mirrorCtx = this._mirror.getContext("2d");
        }
        boxHost.appendChild(this._mirror);
        this._mirrorHost = boxHost;
        this._mirroring = true;
        this.resize();
    }
    // Exit immersive - canvas returns to the box
    exitImmersive(boxHost) {
        this._mirroring = false;
        this._mirror?.parentElement?.removeChild(this._mirror);
        if (boxHost) boxHost.appendChild(this.renderer.domElement);
        this.resize();
    }

    // Replace all layers from App layer defs
    setLayers(defs) {
        this.layers.forEach(l => this._disposeLayerObject(l));
        this.layers = defs.map((d, i) => this._createLayer(d, i));
    }
    addLayer(def) { this.layers.push(this._createLayer(def, this.layers.length)); }
    removeLayer(i) { const l = this.layers[i]; if (l) { this._disposeLayerObject(l); this.layers.splice(i, 1); } }
    setLayerMode(i, style, params) {
        const l = this.layers[i]; if (!l) return;
        l.mode = style;
        l.params = { ...formulaDefaults(style), ...(params || {}) };
        l.smoothParams = null; l.needsRebuild = true;
        l.trackSeed = null; l.sectionShift = null;
        this._deriveMotion(l, i);
    }
    setLayerParam(i, key, value) {
        const l = this.layers[i]; if (!l) return;
        l.params[key] = value;
        if (FORMULAS[l.mode].kind === "custom") l.needsRebuild = true;
    }

    start() {
        if (this.running) return;
        this.running = true;
        this._setupLights();
        this._animate();
    }
    stop() { this.running = false; }

    snapshot() {
        if (this.composer) this.composer.render(); else this.renderer.render(this.scene, this.camera);
        return this.renderer.domElement.toDataURL("image/png");
    }

    // Small JPEG thumbnail for the gallery
    thumbnail(maxW = 480) {
        if (this.composer) this.composer.render(); else this.renderer.render(this.scene, this.camera);
        const src = this.renderer.domElement;
        const scale = Math.min(1, maxW / (src.width || maxW));
        const w = Math.max(1, Math.round(src.width * scale)), h = Math.max(1, Math.round(src.height * scale));
        const c = document.createElement("canvas"); c.width = w; c.height = h;
        c.getContext("2d").drawImage(src, 0, 0, w, h);
        return c.toDataURL("image/jpeg", 0.72);
    }

    /* Layer lifecycle */

    _createLayer(def, idx) {
        const mode = def.style;
        const colors = def.colors || PALETTE_HEX.warm;
        const layer = {
            mode, params: { ...formulaDefaults(mode), ...(def.params || {}) },
            palette: colors.map(h => new THREE.Color(h)),
            smoothParams: null, object: null, geometry: null, morph: null,
            rot: { x: 0, y: 0, z: 0 }, spinDir: 1, spinSpeed: 0.1, swayAmp: 0.2, swayPhase: 0,
            offset: { x: Number(def.offset?.x) || 0, y: Number(def.offset?.y) || 0 },
            needsRebuild: true, rebuildTimer: 0, lastScope: null,
            baseOpacity: 0.8, baseSize: 0.02, baseScale: 1, count: 0, render: null,
            trackSeed: null, sectionShift: null,
        };
        this._deriveMotion(layer, idx);
        return layer;
    }

    _disposeLayerObject(layer) {
        if (!layer.object) return;
        this.scene.remove(layer.object);
        layer.object.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) [].concat(o.material).forEach(m => m.dispose());
        });
        layer.object = null; layer.geometry = null; layer.morph = null;
    }

    _setupLights() {
        if (this.objects.pointLight) return;
        this.scene.add(new THREE.AmbientLight(0x222222, 0.6));
        const pt = new THREE.PointLight(0xffffff, 1.5, 40);
        pt.position.set(3, 3, 6);
        this.scene.add(pt);
        this.objects.pointLight = pt;
    }

    // Motion from the model - roll speed, direction, sway
    _deriveMotion(layer, idx = 0) {
        const c = this.analysis?.cnn || {};
        const mo = c.motion ?? 0.4, e = c.energy ?? 0.4, ar = c.arousal ?? 0.4;
        layer.spinDir = ((c.color_temperature ?? 0.5) >= 0.5 ? 1 : -1) * (idx % 2 ? -1 : 1);
        layer.spinSpeed = 0.05 + mo * 0.45 + e * 0.2;
        layer.swayAmp = 0.1 + ar * 0.22;
        layer.swayPhase = idx * 1.7;
    }

    // Morph rate from the model's motion head
    _deriveMorphRate() {
        this._morphRate = clamp(1.5 + (this.analysis?.cnn?.motion ?? 0.4) * 5, 1.5, 6.5);
    }

    /* Timeline - beat grid + sections */

    // Decaying impulse on each beat of the grid
    _beatEnvelope(t) {
        const beats = this._beats;
        if (!beats.length) return 0;
        if (t < this._beatClock) this._beatIdx = 0;
        this._beatClock = t;
        while (this._beatIdx + 1 < beats.length && beats[this._beatIdx + 1] <= t) this._beatIdx++;
        const last = beats[this._beatIdx];
        if (t < last) return 0;
        const decay = 6 + (this.analysis?.bpm ?? 120) / 30;
        return Math.exp(-decay * (t - last));
    }

    // Angularity reference - the section mean
    _referenceAngularity() {
        return this._trackVec?.angularity ?? this.analysis?.angularity ?? 0.5;
    }

    _segmentIndexAt(t) {
        const segs = this._segments;
        for (let i = segs.length - 1; i >= 0; i--) if (t >= segs[i].start) return i;
        return segs.length ? 0 : -1;
    }

    // Scalars a formula's seed() reads
    static SECTION_KEYS = ["energy", "valence", "arousal", "brightness",
                           "complexity", "motion_speed", "color_temperature"];

    // Advance to the section containing `t`.
    _updateSection(t) {
        const idx = this._segmentIndexAt(t);
        if (idx !== this._segIndex) this._segIndex = idx;
    }

    // Section vector - track character moved by the current section
    _updateSectionVec(dt) {
        const cnn = this.analysis?.cnn || {};
        const seg = this._segments[this._segIndex], ref = this._trackVec;

        const target = {};
        for (const k of Visualizer.SECTION_KEYS) {
            const base = cnn[k] ?? 0.5;
            target[k] = (seg && ref) ? clamp(base + ((seg[k] ?? 0.5) - (ref[k] ?? 0.5)), 0, 1) : base;
        }
        target.motion = target.motion_speed;
        target.angularity = seg?.angularity ?? this._referenceAngularity();

        if (!this._cnnVec) this._cnnVec = { ...target };
        const k = clamp(dt * 0.7, 0, 1);
        for (const key in target) this._cnnVec[key] += (target[key] - this._cnnVec[key]) * k;
        return this._cnnVec;
    }

    // Section delta per parameter
    _sectionShift(layer) {
        const f = FORMULAS[layer.mode], cnn = this.analysis?.cnn;
        if (!f.seed || !cnn || !this._cnnVec) return null;
        if (!layer.trackSeed) {
            layer.trackSeed = f.seed({ cnn, angularity: this._referenceAngularity() });
        }
        const segSeed = f.seed({ cnn: this._cnnVec, angularity: this._cnnVec.angularity });
        const out = layer.sectionShift || (layer.sectionShift = {});
        for (const k in segSeed) out[k] = (segSeed[k] ?? 0) - (layer.trackSeed[k] ?? 0);
        return out;
    }

    /* Per-parameter smoothing */

    _layerScope(layer) {
        const f = FORMULAS[layer.mode], b = this._currentBands, r = this._reactive, out = {};
        const ph = layer.phase || 0;
        const shift = this._sectionShift(layer);
        let i = 0;
        for (const key in f.params) {
            const def = f.params[key];
            const span = def.max - def.min;
            let v = layer.params[key] ?? def.value;
            if (shift) v += (shift[key] ?? 0) * REACT.section;
            if (r && def.audio) {
                v += Math.sin(ph * (0.8 + i * 0.21) + i * 1.7) * span * REACT.wander
                   + (b[def.audio] || 0) * span * REACT.band
                   + this._beat * span * REACT.beat;
            }
            out[key] = clamp(v, def.min, def.max);
            i++;
        }
        return out;
    }

    _smoothLayerParams(layer, dt) {
        const b = this._currentBands;
        if (layer.phase == null) layer.phase = 0;
        if (this._reactive && b.overall > 0.01) layer.phase += dt * (0.15 + b.overall * 1.0);
        const target = this._layerScope(layer);
        if (!layer.smoothParams) layer.smoothParams = { ...target };
        const k = clamp(dt * 4, 0, 1);
        for (const key in target) layer.smoothParams[key] += (target[key] - layer.smoothParams[key]) * k;
    }

    _paramDrift(layer) {
        if (!layer.smoothParams || !layer.lastScope) return Infinity;
        let d = 0;
        for (const k in layer.smoothParams) d += Math.abs(layer.smoothParams[k] - (layer.lastScope[k] ?? layer.smoothParams[k]));
        return d;
    }

    // Palette gradient lookup
    _gradColor(pal, t, target) {
        const n = pal.length;
        const f = clamp(t, 0, 1) * (n - 1), i = Math.floor(f), frac = f - i;
        return target.copy(pal[Math.min(i, n - 1)]).lerp(pal[Math.min(i + 1, n - 1)], frac);
    }

    /* Geometry creation / morph */

    // Position/colour buffers for the vertex count and render type
    _ensureBuffers(layer, count, render) {
        if (layer.object && layer.count === count && layer.render === render) return;
        this._disposeLayerObject(layer);
        const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
        let obj;
        if (render === "points") {
            obj = new THREE.Points(geo, new THREE.PointsMaterial({
                size: 0.02, vertexColors: true, transparent: true, opacity: 0.85,
                sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
            }));
        } else if (render === "line") {
            obj = new THREE.Line(geo, new THREE.LineBasicMaterial({
                vertexColors: true, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending,
            }));
        } else {
            geo.setIndex(new THREE.BufferAttribute(new Uint32Array(count * 4), 1));
            obj = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
                vertexColors: true, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending,
            }));
        }
        layer.object = obj; layer.geometry = geo;
        layer.morph = { current: pos, target: new Float32Array(count * 3) };
        layer.count = count; layer.render = render; layer.firstBuild = true;
        this.scene.add(obj);
    }

    _rebuildLayer(layer) {
        const f = FORMULAS[layer.mode];
        const p = layer.smoothParams || this._layerScope(layer);
        if (f.kind === "custom") this._genWaveform(layer, layer.params);
        else if (f.kind === "attractor") this._genAttractor(layer, f, p);
        else if (layer.mode === "harmonograph") this._genHarmonograph(layer, p);
        else if (layer.mode === "chladni") this._genChladni(layer, p);
        else this._genDeJong(layer, p);
        layer.needsRebuild = false; layer.rebuildTimer = 0; layer.lastScope = { ...p };
        if (layer.morph && (layer.firstBuild || f.snap)) {
            layer.morph.current.set(layer.morph.target);
            layer.geometry.attributes.position.needsUpdate = true;
            layer.firstBuild = false;
        }
    }

    _morphLayer(layer, dt) {
        const cur = layer.morph.current, tgt = layer.morph.target;
        const k = clamp(dt * this._morphRate, 0, 1);
        for (let i = 0; i < cur.length; i++) cur[i] += (tgt[i] - cur[i]) * k;
        layer.geometry.attributes.position.needsUpdate = true;
    }

    /* Generators */

    // Pendulum frequencies, one model head each
    _harmonographConstants(layer) {
        const c = this._cnnVec || this.analysis?.cnn || {};
        const baseF1 = lerp(1.2, 6, c.valence), baseF2 = lerp(1.2, 6, c.arousal);
        const baseF3 = lerp(1.2, 6, c.brightness), baseF4 = lerp(1.2, 6, c.energy);
        const b = this._currentBands, react = this._reactive, beat = this._beat;
        const ph = layer.phase || 0, W = 0.6;
        const tgt = {
            f1: clamp(baseF1 + Math.sin(ph) * W + (react ? b.bass * 1.0 + beat * 0.7 : 0), 1, 8),
            f2: clamp(baseF2 + Math.sin(ph * 0.73 + 1.3) * W + (react ? b.mid * 1.0 + beat * 0.5 : 0), 1, 8),
            f3: clamp(baseF3 + Math.sin(ph * 1.31 + 2.1) * W + (react ? b.high * 0.9 : 0), 1, 8),
            f4: clamp(baseF4 + Math.cos(ph * 0.91 + 0.5) * W + (react ? b.overall * 0.8 + beat * 0.5 : 0), 1, 8),
        };
        if (!layer.hc) layer.hc = { ...tgt };
        for (const key in tgt) layer.hc[key] += (tgt[key] - layer.hc[key]) * 0.34;
        return layer.hc;
    }

    _genHarmonograph(layer, p) {
        const f = FORMULAS.harmonograph; compileFormula(f);
        const N = f.points, span = 40;
        this._ensureBuffers(layer, N, "weave");
        const tgt = layer.morph.target, col = layer.geometry.attributes.color.array;
        const hc = this._harmonographConstants(layer);
        const s = { f1: hc.f1, f2: hc.f2, f3: hc.f3, f4: hc.f4, zt: p.zt, p: p.phase, d: p.damping * 0.02, T: 0 };
        const tmp = new THREE.Color();
        for (let i = 0; i < N; i++) {
            s.T = (i / (N - 1)) * span;
            tgt[i * 3] = f._cx.evaluate(s) * 1.3;
            tgt[i * 3 + 1] = f._cy.evaluate(s) * 1.3;
            tgt[i * 3 + 2] = f._cz.evaluate(s) * 1.3;
            this._gradColor(layer.palette, i / (N - 1), tmp);
            col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
        }
        const off = clamp(Math.round(p.weave), 1, N - 1), idx = layer.geometry.index.array;
        let e = 0;
        for (let i = 0; i < N; i++) {
            idx[e++] = i; idx[e++] = (i + 1) % N;
            idx[e++] = i; idx[e++] = (i + off) % N;
        }
        layer.geometry.index.needsUpdate = true;
        layer.geometry.attributes.color.needsUpdate = true;
        layer.baseOpacity = 0.55; layer.baseScale = 1;
    }

    // de Jong constants a/b/c/d, model-derived
    _deJongConstants(layer) {
        const cnn = this._cnnVec || this.analysis?.cnn || {};
        const comp = cnn.complexity ?? 0.4, mo = cnn.motion ?? 0.5;
        const br = cnn.brightness ?? 0.5, va = cnn.valence ?? 0.5, ar = cnn.arousal ?? 0.5;
        const baseA = 1.4 + comp * 1.2 + (va - 0.5) * 0.6;
        const baseB = 1.6 + comp * 0.8 + (ar - 0.5) * 0.6;
        const baseC = 0.3 + br * 1.4 + (va - 0.5) * 0.3;
        const baseD = 1.2 + mo * 1.0 + (ar - 0.5) * 0.3;
        const b = this._currentBands, r = this._reactive;
        const ph = layer.phase || 0, W = 0.95, beat = this._beat;
        const tgt = {
            a: clamp(baseA + Math.sin(ph) * W + (r ? b.bass * 1.1 + beat * 0.9 : 0), -3, 3),
            b: clamp(baseB + Math.sin(ph * 0.73 + 1.3) * W + (r ? b.mid * 1.1 + beat * 0.7 : 0), -3, 3),
            c: clamp(baseC + Math.sin(ph * 1.31 + 2.1) * W + (r ? b.high * 1.3 : 0), -3, 3),
            d: clamp(baseD + Math.cos(ph * 0.91 + 0.5) * W + (r ? b.overall * 1.1 + beat * 0.7 : 0), -3, 3),
        };
        if (!layer.dj) layer.dj = { ...tgt };
        for (const key in tgt) layer.dj[key] += (tgt[key] - layer.dj[key]) * 0.34;
        return layer.dj;
    }

    // de Jong folded into a centred square, 3D fold + depth shading
    _genDeJong(layer, p) {
        const N = FORMULAS.dejong.points, pal = layer.palette, sc = 1.5;
        this._ensureBuffers(layer, N, "points");
        const tgt = layer.morph.target, col = layer.geometry.attributes.color.array;
        const dj = this._deJongConstants(layer);
        const a = dj.a, b = dj.b, c = dj.c, d = dj.d;
        const bands = this._currentBands, r = this._reactive, PI = Math.PI;
        const sizePulse = 1 + (r ? bands.bass * 0.5 : 0);
        const width = sc * (p.scale ?? 1.25) * sizePulse, height = width;
        const foldReact = p.fold ?? 1.5;
        const fold = 0.5 + foldReact * (r ? bands.mid * 1.4 + bands.overall * 0.7 : 0.5);
        const depth = p.depth;
        const tmp = new THREE.Color();
        let x = 0.1, y = 0.1;
        for (let i = 0; i < N; i++) {
            const _x = Math.sin(a * y) - Math.cos(b * x);
            const _y = Math.sin(c * x) - Math.cos(d * y);
            x = _x; y = _y;
            const ff = Math.sin(fold * _x) * Math.cos(fold * _y);
            const warp = 1 + ff * 0.3;
            const px = width * Math.sin(_x) * Math.cos(PI * _x) * warp;
            const py = height * Math.cos(_y) * Math.sin(PI * _y) * warp;
            const nz = ff * depth;
            tgt[i * 3] = px; tgt[i * 3 + 1] = py; tgt[i * 3 + 2] = nz * sc;
            this._gradColor(pal, clamp(Math.hypot(px, py) / (width * 1.4), 0, 1), tmp);
            const shade = 0.5 + 0.5 * clamp(nz * 1.4 + 0.5, 0, 1);
            col[i * 3] = tmp.r * shade; col[i * 3 + 1] = tmp.g * shade; col[i * 3 + 2] = tmp.b * shade;
        }
        layer.geometry.attributes.color.needsUpdate = true;
        layer.baseOpacity = 0.55; layer.baseSize = 0.009; layer.object.material.size = 0.009;
    }

    // Build the Chladni shader plate
    _genChladni(layer, p) {
        if (!layer.object || layer.render !== "shader") {
            this._disposeLayerObject(layer);
            const uniforms = {
                uM: { value: p.m }, uN: { value: p.n }, uM2: { value: p.m + 2 }, uN2: { value: p.n + 1 },
                uMix: { value: 0.3 }, uWidth: { value: 0.12 }, uWarp: { value: 0.2 },
                uTime: { value: 0 }, uGrain: { value: 0.35 }, uBright: { value: 1 },
                uColA: { value: new THREE.Color() }, uColB: { value: new THREE.Color() },
            };
            const geo = new THREE.PlaneGeometry(4, 4, 1, 1);
            const mesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
                uniforms, vertexShader: CHLADNI_VERT, fragmentShader: CHLADNI_FRAG,
                transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
            }));
            layer.object = mesh; layer.geometry = geo; layer.morph = null;
            layer.uniforms = uniforms; layer.render = "shader"; layer.count = -3;
            this.scene.add(mesh);
        }
        layer.uniforms.uColA.value.copy(layer.palette[0]);
        layer.uniforms.uColB.value.copy(layer.palette[Math.min(2, layer.palette.length - 1)]);
        layer.baseOpacity = 1; layer.baseScale = 1;
    }

    // Plate character, read per frame from the section vector
    _chladniCharacter() {
        const c = this._cnnVec || this.analysis?.cnn || {};
        return {
            warp: 0.15 + (c.complexity ?? 0.4) * 0.7,
            mix: 0.2 + (c.complexity ?? 0.4) * 0.6,
            speed: 0.1 + (c.motion ?? c.energy ?? 0.3) * 0.6,
            grain: 0.2 + (1 - (c.brightness ?? 0.5)) * 0.4,
            bright: 0.8 + (c.energy ?? 0.4) * 0.7,
            offM: 1 + Math.round((c.complexity ?? 0.4) * 3),
            offN: 1 + Math.round((c.brightness ?? 0.5) * 2),
            swing: lerp(1.5, 5, c.complexity ?? 0.4),
            nGain: lerp(0.7, 1.35, c.brightness ?? 0.5),
        };
    }

    // Plate figure - quantised modes, bass horizontal, treble vertical
    _chladniModes(layer, p, b, ch) {
        const r = this._reactive;
        const swing = ch.swing ?? 3, nGain = ch.nGain ?? 1;
        const bass = r ? clamp(b.bass * 1.7 + this._beat * 0.35, 0, 1) : 0.5;
        const high = r ? clamp(b.high * 2.4, 0, 1) : 0.5;
        const mDrive = p.m + (bass - 0.5) * 2 * swing;
        const nDrive = p.n + (high - 0.5) * 2 * swing * nGain;
        if (!layer.cm) layer.cm = { m: clamp(Math.round(mDrive), 1, 14), n: clamp(Math.round(nDrive), 1, 14) };
        if (Math.abs(mDrive - layer.cm.m) > 0.62) layer.cm.m = clamp(Math.round(mDrive), 1, 14);
        if (Math.abs(nDrive - layer.cm.n) > 0.62) layer.cm.n = clamp(Math.round(nDrive), 1, 14);
        return layer.cm;
    }

    // Per-frame uniform update
    _updateChladni(layer, b, dt, t) {
        const u = layer.uniforms; if (!u) return;
        this._smoothLayerParams(layer, dt);
        const p = layer.smoothParams, ch = this._chladniCharacter(), reactive = this._reactive;
        const k = clamp(dt * 6, 0, 1), ez = (uni, tv) => uni.value += (tv - uni.value) * k;
        const md = this._chladniModes(layer, p, b, ch);
        ez(u.uM, md.m); ez(u.uN, md.n);
        ez(u.uM2, md.m + ch.offM); ez(u.uN2, md.n + ch.offN);
        ez(u.uWidth, clamp(0.03 + (p.sharpness / 40) * 0.23, 0.03, 0.26));
        ez(u.uMix, clamp(ch.mix + (reactive ? b.mid * 0.4 : 0), 0, 1));
        ez(u.uWarp, clamp(ch.warp + (reactive ? b.bass * 0.5 : 0), 0, 1.4));
        u.uBright.value += ((ch.bright + (reactive ? b.overall * 0.9 + this._beat * 0.35 : 0)) - u.uBright.value) * clamp(dt * 10, 0, 1);
        u.uGrain.value = ch.grain;
        u.uTime.value += dt * ch.speed * (1 + (reactive ? b.overall : 0));
    }

    // Aizawa silhouette - twist b plus constants e and fc
    _aizawaConstants(layer, p) {
        const cnn = this._cnnVec || this.analysis?.cnn || {};
        const comp = cnn.complexity ?? 0.4, br = cnn.brightness ?? 0.5, ar = cnn.arousal ?? 0.5;
        const b = this._currentBands, r = this._reactive;
        const bass = r ? clamp(b.bass * 1.6, 0, 1) : 0.4;
        const mid  = r ? clamp(b.mid * 1.9, 0, 1) : 0.4;
        const high = r ? clamp(b.high * 2.3, 0, 1) : 0.4;
        const reach = lerp(0.55, 1, ar);
        const tgt = {
            b:  clamp(p.b + (bass - 0.4) * 0.55 * reach, 0.45, 1.0),
            e:  clamp(lerp(0.10, 0.42, comp) + (mid - 0.4) * 0.32 * reach, 0.03, 0.6),
            fc: clamp(lerp(0.02, 0.26, br) + (high - 0.4) * 0.28 * reach, 0, 0.4),
        };
        if (!layer.az) layer.az = { ...tgt };
        for (const key in tgt) layer.az[key] += (tgt[key] - layer.az[key]) * 0.28;
        return layer.az;
    }

    _genAttractor(layer, f, p) {
        const N = f.points;
        this._ensureBuffers(layer, N, "line");
        const tgt = layer.morph.target, col = layer.geometry.attributes.color.array;
        const cp = layer.mode === "aizawa" ? { ...p, ...this._aizawaConstants(layer, p) } : p;
        const s = { x: 0.1, y: 0, z: 0 }, h = cp.speed ?? f.h, sub = f.sub || 1;
        const c = f.center || [0, 0, 0], sc = f.scale;
        for (let i = 0; i < f.warmup; i++) f.step(cp, s, h);
        const tmp = new THREE.Color();
        for (let i = 0; i < N; i++) {
            for (let q = 0; q < sub; q++) f.step(cp, s, h);
            if (!Number.isFinite(s.x + s.y + s.z)) { s.x = 0.1; s.y = 0; s.z = 0; }
            tgt[i * 3] = (s.x - c[0]) * sc; tgt[i * 3 + 1] = (s.y - c[1]) * sc; tgt[i * 3 + 2] = (s.z - c[2]) * sc;
            this._gradColor(layer.palette, i / N, tmp);
            col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
        }
        layer.geometry.attributes.color.needsUpdate = true;
        layer.baseOpacity = 0.7; layer.baseScale = 1;
    }

    /* Waveform - rippling cylindrical mesh */

    _genWaveform(layer, p) {
        this._disposeLayerObject(layer);
        const seg = clamp(Math.round(p.segments), 16, 256), rings = clamp(Math.round(p.rings), 4, 80);
        const length = p.length, radius = p.radius;
        const verts = [], cols = [], idx = [], tmp = new THREE.Color();
        for (let r = 0; r < rings; r++) {
            const z = (r / (rings - 1) - 0.5) * length;
            for (let s = 0; s < seg; s++) {
                const a = (s / seg) * Math.PI * 2;
                verts.push(Math.cos(a) * radius, Math.sin(a) * radius, z);
                this._gradColor(layer.palette, r / (rings - 1), tmp);
                cols.push(tmp.r, tmp.g, tmp.b);
            }
        }
        for (let r = 0; r < rings - 1; r++) for (let s = 0; s < seg; s++) {
            const c = r * seg + s, n = r * seg + (s + 1) % seg;
            idx.push(c, n, c + seg, n, n + seg, c + seg);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
        geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
        geo.setIndex(idx); geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
            vertexColors: true, transparent: true, opacity: 0.7, wireframe: true,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
        }));
        layer.object = mesh; layer.geometry = geo; layer.morph = null;
        layer.wave = { seg, rings, length, radius }; layer.count = -2; layer.render = "waveform";
        layer.baseScale = 1; layer.baseOpacity = 0.7;
        this.scene.add(mesh);
    }

    _updateWaveform(layer, b, dt, t) {
        const mesh = layer.object, w = layer.wave; if (!mesh || !w) return;
        const reactive = this._reactive;
        const bass = reactive ? b.bass : 0.4, mid = reactive ? b.mid : 0.2, high = reactive ? b.high : 0.1, overall = reactive ? b.overall : 0.3;
        const def = FORMULAS.waveform.params.amp, span = def.max - def.min;
        const shift = this._sectionShift(layer);
        const amp = clamp((layer.params.amp ?? 0.7)
            + (shift ? (shift.amp ?? 0) * REACT.section : 0)
            + this._beat * span * REACT.beat, def.min, def.max);
        const pos = mesh.geometry.attributes.position.array, { seg, rings, length, radius } = w;
        for (let r = 0; r < rings; r++) {
            const z = (r / (rings - 1) - 0.5) * length;
            for (let s = 0; s < seg; s++) {
                const a = (s / seg) * Math.PI * 2;
                const rad = radius
                    + Math.sin(a * 3 + t * 2 + r * 0.3) * 0.4 * bass * amp
                    + Math.sin(a * 5 + t * 3 + r * 0.2) * 0.22 * mid * amp
                    + Math.sin(a * 7 + t * 5) * 0.14 * high * amp;
                const i = (r * seg + s) * 3;
                pos[i] = Math.cos(a) * rad; pos[i + 1] = Math.sin(a) * rad;
                pos[i + 2] = z + Math.sin(t + r * 0.5) * 0.25 * overall;
            }
        }
        mesh.geometry.attributes.position.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
    }

    /* Transforms + animation */

    _fillScale() {
        const [halfW, halfH] = this.viewHalfExtent();
        return [(2 * halfW / 4) * 1.06, (2 * halfH / 4) * 1.06];
    }

    // Half the visible world at z=0
    viewHalfExtent() {
        const dist = this.camera.position.length();
        const halfH = Math.tan((this.camera.fov * Math.PI / 180) / 2) * dist;
        return [halfH * this.camera.aspect, halfH];
    }

    // World units per CSS pixel of the drag surface
    worldPerPixel(surfaceHeightPx) {
        const [, halfH] = this.viewHalfExtent();
        return (2 * halfH) / (surfaceHeightPx || this.renderer.domElement.clientHeight || 1);
    }

    // Move a layer, clamped to stay reachable
    setLayerOffset(i, x, y) {
        const l = this.layers[i];
        if (!l) return null;
        const [hw, hh] = this.viewHalfExtent();
        l.offset.x = Math.min(hw, Math.max(-hw, x));
        l.offset.y = Math.min(hh, Math.max(-hh, y));
        return l.offset;
    }

    _applyTransform(layer, f, b, dt, t) {
        const o = layer.object; if (!o) return;
        const reactive = this._reactive;
        if (f.spin === false) {
            if (f.tilt) o.rotation.set(f.tilt[0], f.tilt[1], f.tilt[2]);
            else o.rotation.set(0, 0, 0);
        } else {
            layer.rot.z += layer.spinDir * layer.spinSpeed * (reactive ? 1 + b.overall : 1) * dt;
            const sx = Math.sin(t * 0.25 + layer.swayPhase) * layer.swayAmp;
            const sy = Math.cos(t * 0.19 + layer.swayPhase) * layer.swayAmp;
            o.rotation.set(sx, sy, layer.rot.z);
        }
        if (f.fill) {
            const [fx, fy] = this._fillScale();
            o.scale.set(fx, fy, 1);
            o.position.set(0, 0, 0);
        } else {
            o.scale.setScalar(layer.baseScale * (reactive ? 1 + b.bass * 0.22 : 1));
            o.position.set(layer.offset.x, layer.offset.y, 0);
        }
        const mat = o.material;
        if (mat) {
            mat.opacity = layer.baseOpacity + (reactive ? b.overall * 0.12 : 0);
            if (mat.isPointsMaterial) mat.size = layer.baseSize * (reactive ? 1 + b.high * 0.8 : 1);
        }
    }

    // One simulation step, shared by the live loop and the exporter
    _tick(dt, t, b) {
        this._beat = this._reactive ? this._beatEnvelope(this._trackTime) : 0;
        this._updateSection(this._trackTime);
        this._updateSectionVec(dt);

        for (const layer of this.layers) {
            const f = FORMULAS[layer.mode];
            layer.rebuildTimer += dt;
            const perFrame = f.kind === "custom" || f.kind === "shader";
            if (layer.needsRebuild) {
                this._rebuildLayer(layer);
            } else if (!perFrame) {
                this._smoothLayerParams(layer, dt);
                const thr = f.rebuildThrottle ?? 0.06;
                const live = this._reactive && b.overall > 0.01;
                if (layer.rebuildTimer > thr && (live || this._paramDrift(layer) > 0.004))
                    this._rebuildLayer(layer);
            }
            if (layer.morph && !f.snap) this._morphLayer(layer, dt);
            this._applyTransform(layer, f, b, dt, t);
            if (f.kind === "custom") this._updateWaveform(layer, b, dt, t);
            else if (f.kind === "shader") this._updateChladni(layer, b, dt, t);
        }
        if (this.bloomPass) this.bloomPass.strength = 0.5 + b.overall * 1.1;
    }

    _render() {
        if (this.composer) this.composer.render();
        else this.renderer.render(this.scene, this.camera);
    }

    /* Offline export hooks (exporter.js) */

    // Pause the live loop and switch to export resolution
    exportBegin(w, h) {
        const prev = { running: this.running, pixelRatio: this.renderer.getPixelRatio() };
        this.running = false;
        this.renderer.setPixelRatio(1);
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
        if (this.composer) this.composer.setSize(w, h);
        return prev;
    }

    // Deterministic frame render at time t
    stepFrame(dt, t, bands) {
        this._currentBands = bands;
        this._trackTime = t;
        this._tick(dt, t, bands);
        this._render();
    }

    exportEnd(prev) {
        this.renderer.setPixelRatio(prev.pixelRatio);
        this.resize();
        if (prev.running) { this.running = true; this._animate(); }
    }

    _animate() {
        if (!this.running) return;
        requestAnimationFrame(() => this._animate());
        const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.getElapsedTime(), b = this._currentBands;
        this._tick(dt, t, b);
        this._render();

        if (this._mirroring && this._mirrorHost) {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const cw = this._mirrorHost.clientWidth, ch = this._mirrorHost.clientHeight;
            const src = this.renderer.domElement, sw = src.width, sh = src.height;
            if (cw && ch && sw && sh) {
                const dw = Math.round(cw * dpr), dh = Math.round(ch * dpr);
                if (this._mirror.width !== dw || this._mirror.height !== dh) { this._mirror.width = dw; this._mirror.height = dh; }
                const destAspect = dw / dh, srcAspect = sw / sh;
                let srcW = sw, srcH = sh, sx = 0, sy = 0;
                if (srcAspect > destAspect) { srcW = sh * destAspect; sx = (sw - srcW) / 2; }
                else { srcH = sw / destAspect; sy = (sh - srcH) / 2; }
                this._mirrorCtx.drawImage(src, sx, sy, srcW, srcH, 0, 0, dw, dh);
            }
        }
    }
}

export { Visualizer };
