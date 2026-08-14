// Mic capture worklet: resamples the context rate (usually 48 kHz) down to
// 16 kHz s16le mono and posts fixed 512-sample frames (32 ms, the silero VAD
// chunk size) to the main thread for the WebSocket.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // fractional read position into the backlog
    this.backlog = new Float32Array(0);
    this.frame = new Int16Array(512);
    this.frameIdx = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const merged = new Float32Array(this.backlog.length + input.length);
    merged.set(this.backlog, 0);
    merged.set(input, this.backlog.length);

    // Linear-interpolation resample; always keep one sample of lookahead.
    while (this.pos + 1 < merged.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const s = merged[i] * (1 - frac) + merged[i + 1] * frac;
      this.frame[this.frameIdx++] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
      if (this.frameIdx === this.frame.length) {
        this.port.postMessage(this.frame.buffer.slice(0));
        this.frameIdx = 0;
      }
      this.pos += this.ratio;
    }

    const consumed = Math.floor(this.pos);
    this.backlog = merged.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

registerProcessor('mic', MicProcessor);
