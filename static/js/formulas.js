import { clamp, lerp } from './utils.js';

/* Formula registry - math.js expressions, model-seeded params, band coupling */

const FORMULAS = {
    harmonograph: {
        label: "Harmonograph",
        kind: "expr", render: "weave", spin: true,
        points: 1600, rebuildThrottle: 0.08,
        // Damped 3-axis pendulums, woven net of chords
        exprX: "sin(f1 * T + p) * exp(-d * T) + sin(f2 * T) * exp(-d * T)",
        exprY: "sin(f3 * T + p) * exp(-d * T) + sin(f4 * T) * exp(-d * T)",
        exprZ: "sin(zt * T + p) * exp(-d * T)",
        params: {
            zt: { label: "Depth Twist", min: 1, max: 8, step: 0.01, value: 2.5, audio: "high" },
            weave: { label: "Weave", min: 2, max: 400, step: 1, value: 60 },
            damping: { label: "Fade Out", min: 0.5, max: 6, step: 0.05, value: 2.0 },
            phase: { label: "Phase Shift", min: 0, max: 6.28, step: 0.01, value: 1.57 },
        },
        seed(a) {
            const c = a.cnn;
            return {
                zt: lerp(1.2, 6.5, c.motion),
                weave: Math.round(lerp(20, 240, c.complexity)),
                damping: lerp(0.9, 4.2, 1 - c.arousal),
                phase: lerp(0, 6.28, c.color_temperature),
            };
        },
    },
    chladni: {
        label: "Chladni",
        kind: "shader", spin: false, fill: true,
        // Cymatics plate, fragment shader
        params: {
            m: { label: "Horizontal Waves", min: 1, max: 14, step: 1, value: 4 },
            n: { label: "Vertical Waves", min: 1, max: 14, step: 1, value: 5 },
            sharpness: { label: "Line Width", min: 2, max: 40, step: 0.5, value: 22, audio: "high" },
        },
        // Modes from brightness, line width from angularity
        seed(a) {
            const c = a.cnn, ang = a.angularity ?? 0.5;
            return {
                m: clamp(Math.round(lerp(2, 12, c.complexity)), 1, 14),
                n: clamp(Math.round(lerp(3, 13, c.brightness)), 1, 14),
                sharpness: lerp(34, 8, ang * 0.75 + c.complexity * 0.25),
            };
        },
    },
    dejong: {
        label: "de Jong",
        // Tilted dense point cloud, model-derived curl constants
        kind: "expr", render: "points", spin: false, snap: true, tilt: [-0.26, 0.32, 0],
        points: 90000, rebuildThrottle: 0.06,
        params: {
            scale: { label: "Size / Spread", min: 0.5, max: 2.6, step: 0.01, value: 1.25, audio: "bass" },
            fold: { label: "Fold Reactivity", min: 0, max: 3, step: 0.01, value: 1.5 },
            depth: { label: "3D Depth", min: 0, max: 1.2, step: 0.01, value: 0.5, audio: "bass" },
        },
        // Fold follows spectral sharpness (bouba/kiki)
        seed(a) {
            const c = a.cnn, ang = a.angularity ?? 0.5;
            return {
                scale: lerp(0.9, 2.0, c.energy),
                fold: lerp(0.4, 2.8, ang * 0.7 + c.complexity * 0.3),
                depth: lerp(0.25, 1.0, c.valence),
            };
        },
    },
    waveform: {
        label: "Waveform",
        kind: "custom", spin: true,
        // Rippling cylindrical wireframe
        params: {
            rings: { label: "Rings", min: 6, max: 60, step: 1, value: 28 },
            segments: { label: "Detail", min: 32, max: 200, step: 1, value: 128 },
            radius: { label: "Radius", min: 0.5, max: 2.5, step: 0.05, value: 1.2 },
            length: { label: "Length", min: 2, max: 10, step: 0.1, value: 6 },
            amp: { label: "Wave Height", min: 0, max: 1.5, step: 0.01, value: 0.7, audio: "bass" },
        },
        seed(a) {
            const c = a.cnn;
            return {
                rings: Math.round(lerp(12, 52, c.complexity)),
                segments: Math.round(lerp(64, 190, c.complexity)),
                radius: lerp(0.8, 2.0, c.energy),
                length: lerp(3, 9, c.motion),
                amp: lerp(0.35, 1.35, c.energy),
            };
        },
    },
    lorenz: {
        label: "Lorenz",
        kind: "attractor", render: "line", spin: true,
        points: 12000, warmup: 200, sub: 1, h: 0.0025, scale: 0.07, center: [0, 0, 25],
        rebuildThrottle: 0.04,
        step(p, s, h) {
            const dx = p.sigma * (s.y - s.x), dy = s.x * (p.rho - s.z) - s.y, dz = s.x * s.y - p.beta * s.z;
            s.x += dx * h; s.y += dy * h; s.z += dz * h;
        },
        params: {
            rho: { label: "Flow", min: 10, max: 60, step: 0.1, value: 28, audio: "bass" },
            sigma: { label: "Swirl", min: 4, max: 16, step: 0.1, value: 10, audio: "mid" },
            beta: { label: "Decay", min: 1, max: 5, step: 0.01, value: 2.667 },
            speed: { label: "Speed", min: 0.001, max: 0.005, step: 0.00025, value: 0.0025, audio: "high" },
        },
        seed(a) {
            const c = a.cnn;
            return {
                rho: lerp(14, 55, c.energy),
                sigma: lerp(5, 15, c.arousal),
                beta: lerp(1.5, 4, c.complexity),
                speed: lerp(0.0012, 0.0045, c.motion),
            };
        },
    },
    thomas: {
        label: "Thomas",
        kind: "attractor", render: "line", spin: true,
        points: 12000, warmup: 200, sub: 2, h: 0.04, scale: 0.92, center: [1.35, 1.35, 1.35],
        rebuildThrottle: 0.04,
        step(p, s, h) {
            const dx = Math.sin(s.y) - p.b * s.x, dy = Math.sin(s.z) - p.b * s.y, dz = Math.sin(s.x) - p.b * s.z;
            s.x += dx * h; s.y += dy * h; s.z += dz * h;
        },
        params: {
            b: { label: "Friction", min: 0.10, max: 0.33, step: 0.001, value: 0.208, audio: "bass" },
            speed: { label: "Speed", min: 0.02, max: 0.08, step: 0.001, value: 0.04, audio: "high" },
        },
        seed(a) {
            const c = a.cnn;
            return {
                b: lerp(0.12, 0.30, c.complexity),
                speed: lerp(0.025, 0.07, c.motion),
            };
        },
    },
    aizawa: {
        label: "Aizawa",
        kind: "attractor", render: "line", spin: true,
        points: 14000, warmup: 300, sub: 1, h: 0.01, scale: 1.3, center: [0, 0, 0.3],
        rebuildThrottle: 0.04,
        // e and fc are model-driven; the classic form leaves them fixed
        step(p, s, h) {
            const x = s.x, y = s.y, z = s.z, e = p.e ?? 0.25, fc = p.fc ?? 0.1;
            const dx = (z - p.b) * x - p.d * y;
            const dy = p.d * x + (z - p.b) * y;
            const dz = p.c + p.a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + fc * z * x * x * x;
            s.x += dx * h; s.y += dy * h; s.z += dz * h;
        },
        params: {
            a: { label: "Lift", min: 0.7, max: 1.1, step: 0.005, value: 0.95, audio: "bass" },
            b: { label: "Twist", min: 0.5, max: 0.9, step: 0.005, value: 0.7 },
            c: { label: "Pull", min: 0.3, max: 0.9, step: 0.005, value: 0.6 },
            d: { label: "Spin", min: 2.5, max: 4.5, step: 0.01, value: 3.5, audio: "high" },
        },
        seed(a) {
            const c = a.cnn;
            return {
                a: lerp(0.75, 1.08, c.energy),
                b: lerp(0.55, 0.88, c.valence),
                c: lerp(0.35, 0.85, c.complexity),
                d: lerp(2.6, 4.4, c.arousal),
            };
        },
    },
};

function formulaDefaults(id) {
    const out = {};
    const p = FORMULAS[id].params;
    for (const k in p) out[k] = p[k].value;
    return out;
}

/* Shared reactivity rule - section, wander, band, beat */
const REACT = { section: 1.0, wander: 0.05, band: 0.10, beat: 0.14 };

function compileFormula(f) {
    if (f._compiled) return;
    if (f.expr) f._c = math.compile(f.expr);
    if (f.exprX) f._cx = math.compile(f.exprX);
    if (f.exprY) f._cy = math.compile(f.exprY);
    if (f.exprZ) f._cz = math.compile(f.exprZ);
    f._compiled = true;
}

/* Chladni shader - standing-wave field per pixel */
const CHLADNI_VERT = `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const CHLADNI_FRAG = `
precision highp float;
varying vec2 vUv;
uniform float uM, uN, uM2, uN2, uMix, uWidth, uWarp, uTime, uGrain, uBright;
uniform vec3 uColA, uColB;
#define PI 3.14159265359
float chl(vec2 p, float m, float n) {
    return sin(n * PI * p.x) * sin(m * PI * p.y) - sin(m * PI * p.x) * sin(n * PI * p.y);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
    vec2 p = vUv;
    p += uWarp * vec2(sin(p.y * PI * 3.0 + uTime), cos(p.x * PI * 3.0 + uTime)) * 0.03;
    float v = mix(chl(p, uM, uN), chl(p, uM2, uN2), uMix);
    float line = 1.0 - smoothstep(0.0, uWidth, abs(v));
    line = pow(line, 1.6);
    float g = hash(floor(vUv * 700.0) + floor(uTime * 3.0));
    line *= mix(1.0, 0.5 + 0.5 * g, uGrain);
    line *= uBright;
    vec3 col = mix(uColA, uColB, clamp(distance(vUv, vec2(0.5)) * 1.6, 0.0, 1.0)) * line;
    gl_FragColor = vec4(col, 1.0);
}`;

export { FORMULAS, formulaDefaults, REACT, compileFormula, CHLADNI_VERT, CHLADNI_FRAG };
