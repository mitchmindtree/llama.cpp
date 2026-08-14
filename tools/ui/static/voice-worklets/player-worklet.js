// Playback worklet: a segment-aware ring of Float32 audio queued by the main
// thread. Tracks per-segment samples actually played and reports progress
// (~5x/s and at segment end) so the server can truncate history at what the
// user really heard on barge-in. 'flush' empties the queue on interrupt.
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = []; // { seg, data: Float32Array, idx }
    this.played = new Map(); // seg id -> cumulative samples played
    this.dirty = new Set();
    this.sinceReport = 0;
    this.port.onmessage = (e) => {
      const m = e.data;

      if (m.type === 'audio') {
        this.queue.push({ data: m.samples, idx: 0, seg: m.seg });
      } else if (m.type === 'flush') {
        this.queue = [];
        this.report();
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];

    let i = 0;

    while (i < out.length && this.queue.length > 0) {
      const head = this.queue[0];
      const n = Math.min(out.length - i, head.data.length - head.idx);

      out.set(head.data.subarray(head.idx, head.idx + n), i);
      i += n;
      head.idx += n;
      this.played.set(head.seg, (this.played.get(head.seg) || 0) + n);
      this.dirty.add(head.seg);

      if (head.idx >= head.data.length) {
        this.queue.shift();
        this.report(); // segment boundary: report promptly
      }
    }
    this.sinceReport += out.length;

    if (this.dirty.size > 0 && this.sinceReport >= sampleRate / 5) {
      this.report();
    }

    return true;
  }

  report() {
    for (const seg of this.dirty) {
      this.port.postMessage({ samples: this.played.get(seg), seg });
    }
    this.dirty.clear();
    this.sinceReport = 0;
  }
}

registerProcessor('player', PlayerProcessor);
