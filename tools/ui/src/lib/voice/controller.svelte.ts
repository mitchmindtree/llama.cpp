/**
 * Voice controller singleton: owns the speak-replies pipeline (sentence
 * chunker -> gateway /speech -> player) and the continuous-conversation mode
 * (mic -> gateway WS -> transcripts -> chatStore.sendMessage), including
 * barge-in (flush playback, abort TTS fetches, stop generation).
 *
 * Holds reactive $state only. Effects (watching the chat stream, lifecycle)
 * live in VoiceHost.svelte, since $effect requires a component context.
 */

import { toast } from 'svelte-sonner';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { config } from '$lib/stores/settings.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { SentenceChunker, type Segment } from './chunker';
import { clearDim, scheduleDim, type DimState } from './dimmer';
import * as gateway from './gateway';
import { VoiceMic } from './mic';
import { VoicePlayer } from './player';

interface TurnSegment {
	segId: number;
	seg: Segment;
	sent: number; // samples streamed to the player
}

/** Spoken-progress accounting for one assistant turn. */
interface TurnTracker {
	convId: string;
	msgId: string | null;
	segments: TurnSegment[];
	latex: boolean; // raw offsets unreliable, dim at block granularity
}

export type VoiceStatus = 'idle' | 'listening' | 'transcribing' | 'speaking';

class VoiceController {
	available = $state(false);
	speakEnabled = $state(false);
	micEnabled = $state(false);
	status = $state<VoiceStatus>('idle');
	voices = $state<gateway.VoiceInfo[]>([]);
	/** Message id currently being replayed via the per-message play button. */
	replayingMsgId = $state<string | null>(null);
	/** Voice name currently being previewed from the settings tab. */
	previewingVoice = $state<string | null>(null);
	/** Smoothed mic input level 0..1, for the activity pulse. */
	micLevel = $state(0);

	private player = new VoicePlayer();
	private mic = new VoiceMic();
	private ws: WebSocket | null = null;
	private chunker = new SentenceChunker();

	private newChunker(): SentenceChunker {
		// voiceParagraphChunks off = the earlier one-sentence-per-request mode.
		return new SentenceChunker(config().voiceParagraphChunks !== false);
	}
	private lastLen = 0;
	private queue: TurnSegment[] = [];
	private pumping = false;
	private abort: AbortController | null = null;
	private segCounter = 0;
	private tracker: TurnTracker | null = null;
	private trackerStale = true; // next enqueued segment starts a new turn
	private micLevelRaw = 0;
	private micLevelFlushed = 0;
	private lastAudibleAt = 0;
	// Serializes gateway events (barge-in, transcripts) so a transcript can
	// never land inside stopGeneration's awaited-save window, where upstream
	// would queue it as a pending message and then drop it.
	private ops: Promise<void> = Promise.resolve();

	get micStream() {
		return this.mic.micStream;
	}

	async probe(): Promise<void> {
		this.available = window.isSecureContext && Boolean(config().voiceEnabled) && (await gateway.health());
		if (this.available) {
			await this.refreshVoices();
			// Initial toggle state from settings. Playback still initializes
			// lazily in pump(), after some user gesture has happened.
			this.speakEnabled = Boolean(config().voiceSpeakReplies);
		}
	}

	async refreshVoices(): Promise<void> {
		try {
			this.voices = await gateway.listVoices();
		} catch (err) {
			console.warn('voice: listing voices failed', err);
		}
	}

	async toggleSpeak(): Promise<void> {
		if (!this.speakEnabled) {
			try {
				await this.player.init();
				await this.player.resume(); // inside the click gesture, satisfies autoplay
				this.player.onDrain = () => this.onDrained();
				this.speakEnabled = true;
			} catch (err) {
				toast.error(`Voice playback failed: ${err instanceof Error ? err.message : err}`);
			}
		} else {
			this.speakEnabled = false;
			this.stopSpeaking();
		}
	}

	async toggleMic(): Promise<void> {
		if (!this.micEnabled) {
			try {
				// Conversation mode implies hearing the replies.
				if (!this.speakEnabled) {
					await this.toggleSpeak();
					if (!this.speakEnabled) throw new Error('voice playback unavailable');
				}
				this.openWs();
				await this.mic.start((frame) => {
					this.trackMicLevel(frame);
					// Half-duplex: speaker users without working echo cancellation
					// mute the uplink while the assistant is audible (plus a short
					// hangover) instead of having it barge in on itself.
					if (this.halfDuplexGated()) return;
					if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(frame);
				});
				this.micEnabled = true;
				this.status = 'listening';
			} catch (err) {
				this.mic.stop();
				this.ws?.close();
				this.ws = null;
				const detail = err instanceof Error ? err.message : String(err);
				toast.error(
					/permission|denied|notallowed/i.test(detail)
						? 'Microphone access was denied.'
						: `Voice conversation failed: ${detail}`
				);
			}
		} else {
			this.micEnabled = false;
			this.mic.stop();
			this.ws?.close();
			this.ws = null;
			this.micLevelRaw = 0;
			this.micLevel = 0;
			if (this.status !== 'speaking') this.status = 'idle';
			// Defensive un-stick of the global edit-mode flag (see bargeIn).
			chatStore.clearEditMode();
		}
	}

	private halfDuplexGated(): boolean {
		if (!config().voiceHalfDuplex) return false;
		if (this.status === 'speaking' || this.pumping || this.player.hasPending) {
			this.lastAudibleAt = performance.now();
			return true;
		}
		return performance.now() - this.lastAudibleAt < 400;
	}

	/** Fast-attack slow-decay input level, flushed to $state ~12x/s. */
	private trackMicLevel(frame: ArrayBuffer): void {
		const samples = new Int16Array(frame);
		let sum = 0;
		for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
		const instant = Math.min(1, Math.sqrt(sum / samples.length) / 6000);
		this.micLevelRaw = Math.max(instant, this.micLevelRaw * 0.85);
		const now = performance.now();
		if (now - this.micLevelFlushed >= 80) {
			this.micLevelFlushed = now;
			this.micLevel = Math.round(this.micLevelRaw * 100) / 100;
		}
	}

	/** Feed from VoiceHost's $effect watching chatStore.currentResponse. */
	onStreamedText(full: string): void {
		if (!this.speakEnabled || !this.available) {
			this.lastLen = full.length;
			return;
		}
		if (full.length < this.lastLen) {
			// Stream reset: completion ended (-> '') or an agentic turn started a
			// fresh message. Flush the tail of the previous turn, then re-feed
			// whatever the new stream already holds.
			this.finishTurn();
			this.lastLen = 0;
		}
		const delta = full.slice(this.lastLen);
		this.lastLen = full.length;
		if (delta) {
			// A fresh live turn takes precedence over a running replay.
			if (this.replayingMsgId !== null) this.stopSpeaking();
			if (this.tracker && !this.trackerStale && /\$|\\\(|\\\[/.test(delta)) {
				this.tracker.latex = true;
			}
			for (const seg of this.chunker.push(delta)) this.enqueueSegment(seg);
			this.refreshDim();
		}
	}

	/** Replay a committed assistant message with the current voice, or stop
	 *  the replay when it is already running for this message. */
	toggleReplay(msgId: string, content: string): void {
		if (this.replayingMsgId === msgId) {
			this.stopSpeaking();
			return;
		}
		this.stopSpeaking();
		this.replayingMsgId = msgId;
		const chunker = this.newChunker();
		const segments = chunker.push(content);
		const tail = chunker.flush();
		if (tail) segments.push(tail);
		for (const seg of segments) this.enqueueSegment(seg);
		if (this.tracker && /\$|\\\(|\\\[/.test(content)) this.tracker.latex = true;
		this.refreshDim();
	}

	private lastAssistantMessageId(): string | null {
		const messages = conversationsStore.activeMessages;
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === 'assistant') return messages[i].id;
		}
		return null;
	}

	/** Preview a voice from the settings tab, or stop the running preview.
	 *  Exclusive: starting one stops whatever is currently playing. */
	togglePreview(name: string): void {
		if (this.previewingVoice === name) {
			this.stopSpeaking();
			return;
		}
		this.stopSpeaking();
		this.previewingVoice = name;
		this.enqueueSegment({
			text: `This is the ${name.replaceAll(/[_-]/g, ' ')} voice speaking.`,
			voiceTag: name,
			rawStart: 0,
			rawEnd: 0
		});
	}

	private finishTurn(): void {
		const seg = this.chunker.flush();
		if (seg) this.enqueueSegment(seg);
		this.chunker = this.newChunker();
		// The next turn is a new message: raw offsets restart at zero, so the
		// tracker must rebind on its first segment.
		this.trackerStale = true;
	}

	private resolveVoice(tag: string | null): string {
		if (tag && this.voices.some((v) => v.name === tag)) return tag;
		return String(config().voiceVoice || 'narrator');
	}

	private enqueueSegment(seg: Segment): void {
		if (this.trackerStale) {
			// First spoken segment of a new turn: bind the tracker to the
			// message being streamed, for truncation and dimming.
			const convId = conversationsStore.activeConversation?.id ?? '';
			this.tracker = {
				convId,
				msgId: convId ? (chatStore.getChatStreamingPublic(convId)?.messageId ?? null) : null,
				segments: [],
				latex: /\$|\\\(|\\\[/.test(seg.text)
			};
			this.trackerStale = false;
		}
		const record: TurnSegment = { segId: ++this.segCounter, seg, sent: 0 };
		this.tracker?.segments.push(record);
		this.queue.push(record);
		void this.pump();
	}

	/** Raw-content offset of the last heard word, null when nothing played. */
	private heardRawBoundary(tracker: TurnTracker): number | null {
		let boundary: number | null = null;
		for (const record of tracker.segments) {
			// Zero-span segments (voice previews) carry no source position.
			if (record.seg.rawEnd <= record.seg.rawStart) continue;
			const played = this.player.playedSamples(record.segId);
			if (record.sent > 0 && played >= record.sent) {
				boundary = record.seg.rawEnd;
				continue;
			}
			if (played > 0 && record.sent > 0) {
				const span = record.seg.rawEnd - record.seg.rawStart;
				boundary = record.seg.rawStart + Math.floor((span * played) / record.sent);
			}
			break;
		}
		return boundary;
	}

	// The dimmer pulls current state through this provider on its own
	// rAF-coalesced schedule (playback ticks, stream deltas and renderer
	// mutations all just poke the scheduler). Null clears the highlight.
	// NOTE: no trackerStale here - the tracker outlives generation so the
	// boundary keeps advancing while the tail of the reply is still spoken.
	private dimProvider = (): DimState | null => {
		if (this.tracker === null) return null;
		if (!this.speakEnabled && this.replayingMsgId === null) return null;
		// The dimmer locates the LAST message's content, so dim a replay only
		// when the replayed message actually is the last assistant message.
		if (this.replayingMsgId !== null && this.replayingMsgId !== this.lastAssistantMessageId()) {
			return null;
		}
		if (!this.pumping && this.queue.length === 0 && !this.player.hasPending) return null;
		return {
			boundary: this.heardRawBoundary(this.tracker) ?? 0,
			blockAligned: this.tracker.latex
		};
	};

	private refreshDim(): void {
		scheduleDim(this.dimProvider);
	}

	private async pump(): Promise<void> {
		if (this.pumping) return;
		this.pumping = true;
		this.setSpeaking(true);
		try {
			// Lazy init: submitting a message is a user gesture, so resume()
			// succeeds here even when speak-replies was enabled from settings
			// rather than via the toggle button.
			try {
				await this.player.init();
				this.player.onDrain = () => this.onDrained();
				this.player.onProgress = () => this.refreshDim();
				await this.player.resume();
			} catch (err) {
				toast.error(
					`Voice playback failed: ${err instanceof Error ? err.message : err}`
				);
				this.speakEnabled = false;
				this.queue = [];
				return;
			}
			while (this.queue.length > 0) {
				const record = this.queue.shift()!;
				const voice = this.resolveVoice(record.seg.voiceTag);
				this.abort = new AbortController();
				try {
					for await (const pcm of gateway.speechStream(
						record.seg.text,
						voice,
						this.abort.signal
					)) {
						record.sent += pcm.length;
						this.player.enqueue(record.segId, pcm);
					}
				} catch (err) {
					if (!this.abort.signal.aborted) console.warn('voice: tts segment failed', err);
				}
			}
		} finally {
			this.abort = null;
			this.pumping = false;
			if (!this.player.hasPending) this.onDrained();
		}
	}

	private onDrained(): void {
		if (this.pumping || this.queue.length > 0) return;
		clearDim();
		this.replayingMsgId = null;
		this.previewingVoice = null;
		this.setSpeaking(false);
	}

	private setSpeaking(speaking: boolean): void {
		if (speaking) {
			this.status = 'speaking';
			this.sendWs({ type: 'mode', value: 'speaking' });
		} else {
			this.status = this.micEnabled ? 'listening' : 'idle';
			this.sendWs({ type: 'mode', value: 'listening' });
		}
	}

	stopSpeaking(): void {
		this.queue = [];
		this.abort?.abort();
		this.player.flush();
		this.chunker = this.newChunker();
		// Kill the turn outright so late playback ticks cannot dim a dead
		// message. Barge-in snapshots the tracker before calling this.
		this.tracker = null;
		this.trackerStale = true;
		this.replayingMsgId = null;
		this.previewingVoice = null;
		clearDim();
		this.setSpeaking(false);
	}

	/** Chain an operation so gateway events run strictly one at a time. */
	private enqueueOp(fn: () => Promise<void>): void {
		this.ops = this.ops.then(fn).catch((err) => {
			console.warn('voice: operation failed', err);
		});
	}

	private bargeIn(): void {
		if (!config().voiceBargeIn) return;
		if (this.status !== 'speaking' && !chatStore.isLoading) return;
		// Snapshot the spoken boundary before flushing settles the accounts.
		// Replay trackers carry offsets into an old message, never truncate.
		const replaying = this.replayingMsgId !== null;
		const tracker = this.tracker;
		const boundary = tracker ? this.heardRawBoundary(tracker) : null;
		this.stopSpeaking();
		this.enqueueOp(async () => {
			await chatStore.stopGeneration();
			if (!replaying && tracker !== null) await this.truncateToHeard(tracker, boundary);
			// The pending-message bubble shares a global edit-mode flag with the
			// message edit forms; destroying it via stopGeneration can orphan
			// the flag and gray out the prompt input. Clearing is idempotent.
			chatStore.clearEditMode();
		});
	}

	/** Rewrite the interrupted assistant message to what was actually heard. */
	private async truncateToHeard(tracker: TurnTracker, boundary: number | null): Promise<void> {
		if (tracker.msgId === null || boundary === null) return;
		const idx = conversationsStore.findMessageIndex(tracker.msgId);
		if (idx < 0) return;
		const content = conversationsStore.activeMessages[idx]?.content ?? '';
		if (boundary >= content.length) return;
		let cut = boundary;
		// Never split a word at the dim boundary estimate.
		while (cut > 0 && cut < content.length && !/\s/.test(content[cut - 1])) cut -= 1;
		if (cut === 0) return;
		const truncated = content.slice(0, cut).trimEnd() + '...';
		// The onAssistantTurnComplete pattern: field-wise store update for the
		// UI and the next request, Dexie for persistence, currNode so the
		// message stays on the active branch (never refreshActiveMessages here,
		// the in-memory currNode still points at the user message after abort).
		conversationsStore.updateMessageAtIndex(idx, { content: truncated });
		await DatabaseService.updateMessage(tracker.msgId, { content: truncated });
		await conversationsStore.updateCurrentNode(tracker.msgId);
	}

	private openWs(): void {
		const ws = gateway.openWs();
		ws.onmessage = (e) => {
			if (typeof e.data !== 'string') return;
			const msg = JSON.parse(e.data);
			switch (msg.type) {
				case 'speech_start':
					this.bargeIn();
					break;
				case 'state':
					if (this.status !== 'speaking') {
						this.status = msg.value === 'transcribing' ? 'transcribing' : 'listening';
					}
					break;
				case 'transcript':
					this.enqueueOp(async () => {
						// Never queue a transcript into upstream's pending-message
						// machinery (which materializes an edit-form bubble and can
						// be dropped by a concurrent stop): settle any in-flight
						// turn first, then send. stopGeneration is a no-op when
						// idle.
						await chatStore.stopGeneration();
						await chatStore.sendMessage(msg.text);
					});
					break;
				case 'error':
					toast.error(`Voice gateway: ${msg.message}`);
					break;
			}
		};
		ws.onclose = () => {
			if (this.micEnabled) {
				// The gateway or proxy dropped us: retire the mic cleanly so the
				// toggle reflects reality instead of silently eating audio.
				void this.toggleMic();
			}
		};
		this.ws = ws;
	}

	private sendWs(msg: object): void {
		if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
	}
}

export const voiceController = new VoiceController();
