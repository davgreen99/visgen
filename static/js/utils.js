const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

// Map a 0..1 model head onto a parameter range
const lerp = (lo, hi, t) => lo + (hi - lo) * clamp(t ?? 0.5, 0, 1);

/* Model palette - Palmer et al. (2013) */
function aiPalette(a) {
    const c = a.cnn || {};
    const t = c.color_temperature ?? 0.5, e = c.energy ?? 0.5, ar = c.arousal ?? 0.5;
    const va = c.valence ?? 0.5, br = c.brightness ?? 0.5, cx = c.complexity ?? 0.5;

    const vaN = clamp((va - 0.30) / 0.35, 0, 1);
    const tN = clamp((t - 0.08) / 0.62, 0, 1);
    const warmth = 0.35 * vaN + 0.65 * tN;
    const hue = 0.66 - warmth * 0.58;
    const fan = lerp(0.10, 0.30, cx);
    const sat = lerp(0.38, 0.95, ar * 0.75 + e * 0.25);
    const mid = lerp(0.26, 0.68, vaN * 0.6 + br * 0.4);
    const spread = lerp(0.13, 0.28, e);

    return [0, 1, 2, 3].map(i => {
        const k = i / 3;
        return new THREE.Color().setHSL(
            hue + fan * (k - 0.5),
            clamp(sat - Math.abs(k - 0.5) * 0.15, 0.2, 1),
            clamp(mid + (k - 0.5) * spread, 0.12, 0.9),
        );
    });
}

// Hex presets for the colour pickers
const PALETTE_HEX = {
    warm: ["#d42b2b", "#ff6b35", "#f7c948", "#ff3d6e"],
    cool: ["#3b82f6", "#8b5cf6", "#06b6d4", "#6366f1"],
    mono: ["#ffffff", "#cccccc", "#999999", "#e8e8e8"],
};
function aiPaletteHex(a) { return aiPalette(a).map(c => "#" + c.getHexString()); }

export { $, $$, clamp, lerp, aiPalette, PALETTE_HEX, aiPaletteHex };
