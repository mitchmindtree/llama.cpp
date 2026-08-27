/**
 * Client for the voice gateway (nixos/voice-chat.nix, pkgs/voice-chat-server).
 *
 * The base URL comes from the voiceGatewayUrl setting (default "voice") and
 * resolves relative to the page origin: behind the tailscale serve HTTPS
 * front the gateway is mounted at /voice on the same origin, so requests are
 * same-origin and need no CORS. On the plain :8080 origin the health probe
 * fails and voice stays disabled.
 */

import { settingsStore } from '$lib/stores';

export interface VoiceInfo {
	name: string;
	ref_text: string;
}

function base(): string {
	let raw = String(settingsStore.config.voiceGatewayUrl || 'voice');

	if (!raw.endsWith('/')) raw += '/';

	return new URL(raw, document.baseURI).toString();
}

export async function health(): Promise<boolean> {
	try {
		const r = await fetch(new URL('health', base()), { signal: AbortSignal.timeout(3000) });

		return r.ok && (await r.json()).ok === true;
	} catch {
		return false;
	}
}

export async function listVoices(): Promise<VoiceInfo[]> {
	const r = await fetch(new URL('voices', base()));

	if (!r.ok) throw new Error(`listing voices failed: ${r.status}`);

	return r.json();
}

export async function cloneVoice(form: FormData): Promise<{ name: string; transcript: string }> {
	const r = await fetch(new URL('voices', base()), { body: form, method: 'POST' });

	if (!r.ok) throw new Error((await r.text()).slice(0, 300));

	return r.json();
}

export async function deleteVoice(name: string): Promise<void> {
	const r = await fetch(new URL(`voices/${encodeURIComponent(name)}`, base()), {
		method: 'DELETE'
	});

	if (!r.ok) throw new Error(`deleting ${name} failed: ${r.status}`);
}

export async function design(body: {
	text: string;
	instructions: string;
	seed?: number;
}): Promise<Blob> {
	const r = await fetch(new URL('design', base()), {
		body: JSON.stringify(body),
		headers: { 'Content-Type': 'application/json' },
		method: 'POST'
	});

	if (!r.ok) throw new Error((await r.text()).slice(0, 300));

	return r.blob();
}

/**
 * Stream synthesized speech as s16le 24 kHz PCM chunks. Chunks are re-aligned
 * to even byte counts so they can be viewed as Int16Array.
 */
export async function* speechStream(
	input: string,
	voice: string,
	signal: AbortSignal
): AsyncGenerator<Int16Array> {
	const body: Record<string, unknown> = { input, response_format: 'pcm', voice };
	// The talker temperature is the prosody dial (0 = flattest/greedy, model
	// default 0.9). The sub-talker sampling fields are dead in the engine.
	const temperature = Number(settingsStore.config.voiceTemperature);

	if (Number.isFinite(temperature) && temperature >= 0) body.temperature = temperature;

	const r = await fetch(new URL('speech', base()), {
		body: JSON.stringify(body),
		headers: { 'Content-Type': 'application/json' },
		method: 'POST',
		signal
	});

	if (!r.ok || !r.body)
		throw new Error(`tts failed: ${r.status} ${(await r.text()).slice(0, 300)}`);

	const reader = r.body.getReader();

	let carry = new Uint8Array(0);

	for (;;) {
		const { done, value } = await reader.read();

		if (done) break;

		let bytes = value;

		if (carry.length > 0) {
			const merged = new Uint8Array(carry.length + bytes.length);

			merged.set(carry, 0);
			merged.set(bytes, carry.length);
			bytes = merged;
			carry = new Uint8Array(0);
		}

		const even = bytes.length - (bytes.length % 2);

		if (even < bytes.length) carry = bytes.slice(even);

		if (even === 0) continue;

		// Copy to guarantee 2-byte alignment for the Int16Array view.
		const aligned = bytes.slice(0, even);

		yield new Int16Array(aligned.buffer, 0, even / 2);
	}
}

export function openWs(): WebSocket {
	const url = new URL('ws', base());

	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';

	return new WebSocket(url);
}
