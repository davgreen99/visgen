/* Offline video exporter - WebCodecs + mp4-muxer, deterministic */

const FPS = 30, WIDTH = 1920, HEIGHT = 1080, SAMPLE_RATE = 48000;

function exportSupported() {
    return typeof VideoEncoder !== "undefined"
        && typeof AudioEncoder !== "undefined"
        && typeof VideoFrame !== "undefined"
        && typeof Mp4Muxer !== "undefined";
}

// Yield without setTimeout - hidden tabs throttle timers
const _yieldChannel = new MessageChannel();
const _yieldWaiters = [];
_yieldChannel.port1.onmessage = () => _yieldWaiters.shift()?.();
function yieldTask() {
    return new Promise(r => { _yieldWaiters.push(r); _yieldChannel.port2.postMessage(0); });
}

// Wait for the encoder queue to drain below `max`
async function drain(encoder, max) {
    while (encoder.encodeQueueSize > max) {
        if ("ondequeue" in encoder) await new Promise(r => encoder.addEventListener("dequeue", r, { once: true }));
        else await yieldTask();
    }
}

// Decode the uploaded file
async function decodeTrack(file) {
    const raw = await file.arrayBuffer();
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE, SAMPLE_RATE);
    return ctx.decodeAudioData(raw);
}

// Per-frame bands from an offline analyser
async function computeBandTimeline(buffer, frames) {
    const length = Math.ceil(buffer.duration * SAMPLE_RATE);
    const ctx = new OfflineAudioContext(buffer.numberOfChannels, length, SAMPLE_RATE);
    const src = ctx.createBufferSource(); src.buffer = buffer;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0.8;
    src.connect(analyser); analyser.connect(ctx.destination);

    const data = new Uint8Array(analyser.frequencyBinCount);
    const bands = new Array(frames);
    const read = () => {
        analyser.getByteFrequencyData(data);
        const len = data.length, bEnd = Math.floor(len * 0.1), mEnd = Math.floor(len * 0.5);
        let bass = 0, mid = 0, high = 0;
        for (let i = 0; i < bEnd; i++) bass += data[i];
        for (let i = bEnd; i < mEnd; i++) mid += data[i];
        for (let i = mEnd; i < len; i++) high += data[i];
        bass /= bEnd * 255; mid /= (mEnd - bEnd) * 255; high /= (len - mEnd) * 255;
        return { bass, mid, high, overall: (bass + mid + high) / 3 };
    };

    for (let i = 1; i < frames; i++) {
        const t = i / FPS;
        if (t >= buffer.duration) break;
        ctx.suspend(t).then(() => { bands[i] = read(); ctx.resume(); });
    }
    src.start();
    await ctx.startRendering();

    const silent = { bass: 0, mid: 0, high: 0, overall: 0 };
    for (let i = 0; i < frames; i++) if (!bands[i]) bands[i] = bands[i - 1] || silent;
    return bands;
}

// Encode PCM in 1-second planar chunks
function encodeAudio(encoder, buffer) {
    const ch0 = buffer.getChannelData(0);
    const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : ch0;
    for (let off = 0; off < ch0.length; off += SAMPLE_RATE) {
        const len = Math.min(SAMPLE_RATE, ch0.length - off);
        const pcm = new Float32Array(len * 2);
        pcm.set(ch0.subarray(off, off + len), 0);
        pcm.set(ch1.subarray(off, off + len), len);
        encoder.encode(new AudioData({
            format: "f32-planar", sampleRate: SAMPLE_RATE, numberOfChannels: 2,
            numberOfFrames: len, timestamp: (off / SAMPLE_RATE) * 1e6, data: pcm,
        }));
    }
}

// Render + encode - returns an MP4 Blob, or null when cancelled
async function exportVideo({ visualizer, file, onProgress, isCancelled }) {
    onProgress("Decoding audio…", 0);
    const audio = await decodeTrack(file);
    const frames = Math.ceil(audio.duration * FPS);

    onProgress("Analyzing audio…", 0);
    const bands = await computeBandTimeline(audio, frames);

    const aac = await AudioEncoder.isConfigSupported({
        codec: "mp4a.40.2", sampleRate: SAMPLE_RATE, numberOfChannels: 2, bitrate: 128000,
    }).then(r => r.supported).catch(() => false);

    const { Muxer, ArrayBufferTarget } = Mp4Muxer;
    const muxer = new Muxer({
        target: new ArrayBufferTarget(),
        video: { codec: "avc", width: WIDTH, height: HEIGHT },
        audio: { codec: aac ? "aac" : "opus", sampleRate: SAMPLE_RATE, numberOfChannels: 2 },
        fastStart: "in-memory",
    });

    const venc = new VideoEncoder({
        output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
        error: e => console.error("VideoEncoder:", e),
    });
    venc.configure({
        codec: "avc1.640028", width: WIDTH, height: HEIGHT,
        bitrate: 12_000_000, framerate: FPS, latencyMode: "quality",
    });

    const aenc = new AudioEncoder({
        output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
        error: e => console.error("AudioEncoder:", e),
    });
    aenc.configure({ codec: aac ? "mp4a.40.2" : "opus", sampleRate: SAMPLE_RATE, numberOfChannels: 2, bitrate: 128000 });
    encodeAudio(aenc, audio);

    const prev = visualizer.exportBegin(WIDTH, HEIGHT);
    try {
        for (let i = 0; i < frames; i++) {
            if (isCancelled()) { venc.close(); aenc.close(); return null; }
            visualizer.stepFrame(1 / FPS, i / FPS, bands[i]);
            const frame = new VideoFrame(visualizer.renderer.domElement, {
                timestamp: (i / FPS) * 1e6, duration: 1e6 / FPS,
            });
            venc.encode(frame, { keyFrame: i % (FPS * 2) === 0 });
            frame.close();
            await drain(venc, 8);
            if (i % 15 === 0) {
                onProgress(`Rendering offline… ${Math.round((i / frames) * 100)}%`, i / frames);
                await yieldTask();
            }
        }
    } finally {
        visualizer.exportEnd(prev);
    }

    onProgress("Finishing…", 1);
    await venc.flush(); await aenc.flush();
    venc.close(); aenc.close();
    muxer.finalize();
    return new Blob([muxer.target.buffer], { type: "video/mp4" });
}

export { exportVideo, exportSupported };
