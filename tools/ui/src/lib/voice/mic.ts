/**
 * Mic capture for the continuous-conversation mode: getUserMedia with echo
 * cancellation plus the mic worklet (static/voice/mic-worklet.js), which
 * resamples the context rate down to 16 kHz s16le and posts 512-sample
 * frames (the silero VAD chunk size) for the gateway WebSocket.
 */

export class VoiceMic {
	private ctx: AudioContext | null = null;
	private stream: MediaStream | null = null;

	get micStream(): MediaStream | null {
		return this.stream;
	}

	async start(onFrame: (frame: ArrayBuffer) => void): Promise<void> {
		if (this.ctx) return;

		this.stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				autoGainControl: true,
				channelCount: 1,
				echoCancellation: true,
				noiseSuppression: true
			}
		});
		this.ctx = new AudioContext();
		// voice-worklets/, not voice/: /voice is the gateway's reverse-proxy
		// mount on the HTTPS origin and would shadow the asset with a 404.
		const url = new URL('voice-worklets/mic-worklet.js', document.baseURI);

		await this.ctx.audioWorklet.addModule(url.toString());
		const node = new AudioWorkletNode(this.ctx, 'mic');

		this.ctx.createMediaStreamSource(this.stream).connect(node);
		node.port.onmessage = (e) => onFrame(e.data as ArrayBuffer);
	}

	stop(): void {
		this.stream?.getTracks().forEach((t) => t.stop());
		this.stream = null;
		void this.ctx?.close();
		this.ctx = null;
	}
}
