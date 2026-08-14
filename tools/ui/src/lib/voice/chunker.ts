/**
 * Incremental sentence chunking of the streamed assistant text for TTS.
 *
 * Emits a small first segment as soon as a sentence completes (low
 * time-to-first-audio) and larger ones after (better prosody, fewer
 * requests). Strips markdown and emoji so the TTS never reads syntax aloud.
 * A segment may carry a voice tag when its text starts with [voice:tag]
 * (multi-voice personas, e.g. a DM voicing NPCs).
 */

export interface Segment {
	text: string;
	voiceTag: string | null;
	/** Span of this segment in the raw streamed text (cumulative offsets).
	 *  Message content renders from the same raw text, so these offsets drive
	 *  both barge-in truncation and the spoken-progress dimming boundary. */
	rawStart: number;
	rawEnd: number;
}

const SENTENCE_END = /[.!?…]+["')\]]*(?:\s|$)/g;
const VOICE_TAG = /^\s*\[voice:([A-Za-z0-9_-]+)\]\s*/;
const VOICE_TAG_ANY = /\[voice:[A-Za-z0-9_-]+\]/;
const ABBREVIATIONS = new Set([
	'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc',
	'e.g', 'i.e', 'no', 'vol', 'approx', 'min', 'max'
]);
const MD_SUBS: Array<[RegExp, string]> = [
	[/```[\s\S]*?(```|$)/g, ' '], // never read code aloud
	[/`([^`]*)`/g, '$1'],
	[/\[([^\]]+)\]\([^)]*\)/g, '$1'], // links -> link text
	[/^[ \t]*[-*+][ \t]+/gm, ''], // bullets
	[/[*_#>|~]+/g, ' '],
	[/\p{Extended_Pictographic}/gu, ''],
	[/[ \t]+/g, ' ']
];

// The TTS keeps no prosody state between requests (each one restarts from the
// voice reference), so bigger chunks read with far better flow - and for ICL
// voices, time-to-first-audio depends on the reference, not the input length.
// Paragraph mode (default): tiny first segment for fast audio start, then
// accumulate complete sentences to paragraph scale. Sentence mode
// (voiceParagraphChunks off): one sentence per request, the earlier behavior.
export const FIRST_MIN_CHARS = 12;
export const SENTENCE_MIN_CHARS = 40; // sentence mode: min segment length
export const SENTENCE_MAX_CHARS = 300; // sentence mode: force-flush ceiling
export const TARGET_CHARS = 400; // paragraph mode: accumulate up to here
export const PARA_MIN_CHARS = 40; // floor for cutting at a paragraph break
export const MAX_CHARS = 700; // paragraph mode: force-flush ceiling

export function clean(text: string): string {
	for (const [pattern, replacement] of MD_SUBS) {
		text = text.replace(pattern, replacement);
	}
	return text.trim();
}

export class SentenceChunker {
	private buf = '';
	private emitted = 0;
	private consumed = 0; // raw chars cut from the stream so far

	constructor(private paragraphMode = true) {}

	push(delta: string): Segment[] {
		this.buf += delta;
		const out: Segment[] = [];
		for (;;) {
			const seg = this.tryCut();
			if (seg === null) break;
			out.push(seg);
		}
		return out;
	}

	flush(): Segment | null {
		const text = this.buf;
		const rawStart = this.consumed;
		this.consumed += text.length;
		this.buf = '';
		return this.emit(text, rawStart, this.consumed);
	}

	private emit(raw: string, rawStart: number, rawEnd: number): Segment | null {
		let voiceTag: string | null = null;
		const m = VOICE_TAG.exec(raw);
		if (m) {
			voiceTag = m[1];
			raw = raw.slice(m[0].length);
		}
		const text = clean(raw);
		if (!text) return null;
		this.emitted += 1;
		return { text, voiceTag, rawStart, rawEnd };
	}

	private tryCut(): Segment | null {
		const first = this.emitted === 0;
		let cut: number | null = null;
		if (first) {
			// Fast first audio: cut at the first complete sentence.
			cut = this.sentenceEnds().find((end) => end >= FIRST_MIN_CHARS) ?? null;
		} else if (this.paragraphMode) {
			// Paragraph scale: cut at the LAST complete sentence, only once the
			// accumulated complete sentences reach the target.
			const last = this.sentenceEnds().at(-1) ?? null;
			if (last !== null && last >= TARGET_CHARS) cut = last;
		} else {
			// Sentence mode: cut at every complete sentence past the minimum.
			cut = this.sentenceEnds().find((end) => end >= SENTENCE_MIN_CHARS) ?? null;
		}
		// Paragraph breaks are natural chunk boundaries; earliest wins.
		const para = this.buf.indexOf('\n\n');
		const paraFloor = first ? FIRST_MIN_CHARS : PARA_MIN_CHARS;
		if (para >= 0 && para + 2 >= paraFloor && (cut === null || para + 2 < cut)) {
			cut = para + 2;
		}
		// An embedded [voice:tag] must start a segment (it is a voice switch),
		// so cut right before it regardless of the minimum length.
		const tag = VOICE_TAG_ANY.exec(this.buf);
		if (tag && this.buf.slice(0, tag.index).trim() && (cut === null || tag.index < cut)) {
			cut = tag.index;
		}
		const maxChars = this.paragraphMode ? MAX_CHARS : SENTENCE_MAX_CHARS;
		if (cut === null && this.buf.length > maxChars) {
			// Prefer a sentence end even below target, else break on punctuation.
			const last = this.sentenceEnds().at(-1) ?? null;
			if (last !== null) {
				cut = last;
			} else {
				const head = this.buf.slice(0, maxChars);
				const fallback = Math.max(head.lastIndexOf(', '), head.lastIndexOf(' '));
				if (fallback > FIRST_MIN_CHARS) cut = fallback + 1;
			}
		}
		return cut === null ? null : this.cutAt(cut);
	}

	/** All guarded sentence-end offsets currently in the buffer, in order. */
	private sentenceEnds(): number[] {
		const ends: number[] = [];
		SENTENCE_END.lastIndex = 0;
		let m: RegExpExecArray | null;
		while ((m = SENTENCE_END.exec(this.buf)) !== null) {
			const end = m.index + m[0].length;
			// Trailing match with no following whitespace yet: the sentence may
			// not be finished ("3." awaiting "5"), wait for more input.
			if (end >= this.buf.length && !/\s/.test(this.buf[end - 1])) break;
			const before = this.buf.slice(0, m.index + 1);
			const trimmed = before.slice(0, -1).trimEnd();
			const word = (trimmed.split(/[\s([]/).pop() ?? '').toLowerCase().replace(/\.+$/, '');
			if (ABBREVIATIONS.has(word)) continue;
			// Decimal / version numbers: digit on both sides of the dot.
			const after = end < this.buf.length ? this.buf.slice(end) : '';
			if (before.endsWith('.') && /\d$/.test(trimmed) && /^\d/.test(after)) continue;
			ends.push(end);
		}
		return ends;
	}

	private cutAt(end: number): Segment | null {
		const raw = this.buf.slice(0, end);
		const rawStart = this.consumed;
		this.consumed += end;
		this.buf = this.buf.slice(end);
		const seg = this.emit(raw, rawStart, this.consumed);
		// Nothing speakable in the cut (pure markdown); keep scanning.
		return seg ?? this.tryCut();
	}
}
