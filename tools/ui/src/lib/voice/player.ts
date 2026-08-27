/**
 * Segment-aware PCM playback: a 24 kHz AudioContext with the player worklet
 * (static/voice/player-worklet.js, a queue that reports played samples per
 * segment). Tracks enqueued vs played totals so the controller knows when
 * audio is still audible (barge-in arming, mode reporting) and when the
 * queue has drained.
 */

export class VoicePlayer {
	onDrain: (() => void) | null = null;
	/** Fires on every playback progress report (~5x/s and at segment ends). */
	onProgress: (() => void) | null = null;
	// Kept referenced so the loopback survives garbage collection.
	private aec: {
		pc1: RTCPeerConnection;
		pc2: RTCPeerConnection;
		audio: HTMLAudioElement;
	} | null = null;
	private ctx: AudioContext | null = null;
	private enqueued = 0;
	private node: AudioWorkletNode | null = null;
	private played = 0;
	private playedPerSegment = new Map<number, number>();

	get hasPending(): boolean {
		return this.enqueued > this.played;
	}

	enqueue(seg: number, pcm: Int16Array): void {
		if (!this.node) return;

		const f32 = Float32Array.from(pcm, (s) => s / 32768);

		this.enqueued += f32.length;
		this.node.port.postMessage({ samples: f32, seg, type: 'audio' }, [f32.buffer]);
	}

	flush(): void {
		this.node?.port.postMessage({ type: 'flush' });
		// Whatever was queued will never play: settle the accounts.
		this.played = this.enqueued;
	}

	async init(): Promise<void> {
		if (this.ctx) return;

		this.ctx = new AudioContext({ sampleRate: 24000 });
		// voice-worklets/, not voice/: /voice is the gateway's reverse-proxy
		// mount on the HTTPS origin and would shadow the asset with a 404.
		const url = new URL('voice-worklets/player-worklet.js', document.baseURI);

		await this.ctx.audioWorklet.addModule(url.toString());
		this.node = new AudioWorkletNode(this.ctx, 'player', { outputChannelCount: [1] });
		try {
			// Browser AEC only subtracts audio it recognizes as WebRTC playback,
			// so TTS through a plain AudioContext leaks from speakers into the
			// mic and self-interrupts the assistant. Looping the output through
			// a local RTCPeerConnection pair and playing the "remote" stream
			// makes the mic's echoCancellation cancel it like any call audio.
			await this.connectViaAecLoopback(this.node);
		} catch (err) {
			console.warn('voice: AEC loopback unavailable, using direct output', err);
			this.node.connect(this.ctx.destination);
		}
		this.node.port.onmessage = (e) => {
			const { samples, seg } = e.data as { seg: number; samples: number };
			const prev = this.playedPerSegment.get(seg) ?? 0;

			if (samples > prev) {
				this.played += samples - prev;
				this.playedPerSegment.set(seg, samples);
			}

			this.onProgress?.();

			if (!this.hasPending) this.onDrain?.();
		};
	}

	playedSamples(seg: number): number {
		return this.playedPerSegment.get(seg) ?? 0;
	}

	/** Must be called from a user gesture at least once (autoplay policy). */
	async resume(): Promise<void> {
		await this.ctx?.resume();
		await this.aec?.audio.play().catch(() => {});
	}

	private async connectViaAecLoopback(node: AudioWorkletNode): Promise<void> {
		const ctx = this.ctx!;
		const dest = ctx.createMediaStreamDestination();

		node.connect(dest);
		const pc1 = new RTCPeerConnection();
		const pc2 = new RTCPeerConnection();

		pc1.onicecandidate = (e) => {
			if (e.candidate) void pc2.addIceCandidate(e.candidate);
		};
		pc2.onicecandidate = (e) => {
			if (e.candidate) void pc1.addIceCandidate(e.candidate);
		};
		for (const track of dest.stream.getTracks()) pc1.addTrack(track, dest.stream);
		const remote = new Promise<MediaStream>((resolve, reject) => {
			pc2.ontrack = (e) => resolve(e.streams[0]);
			setTimeout(() => reject(new Error('loopback negotiation timed out')), 3000);
		});

		await pc1.setLocalDescription(await pc1.createOffer());
		await pc2.setRemoteDescription(pc1.localDescription!);
		await pc2.setLocalDescription(await pc2.createAnswer());
		await pc1.setRemoteDescription(pc2.localDescription!);
		const audio = new Audio();

		audio.srcObject = await remote;
		audio.autoplay = true;
		this.aec = { audio, pc1, pc2 };
		await audio.play();
	}
}
