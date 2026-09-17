import { AttachmentType, ToolResultSegmentKind } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types';
import { groupToolResultLines, parseToolResultWithMedia } from '$lib/utils/agentic';
import { describe, expect, it } from 'vitest';

const audio: DatabaseMessageExtra = {
	base64Data: 'QUJD',
	mimeType: 'audio/wav',
	name: 'mcp-attachment-1-0.wav',
	type: AttachmentType.AUDIO
};
const image: DatabaseMessageExtra = {
	base64Url: 'data:image/png;base64,QUJD',
	name: 'mcp-attachment-1-1.png',
	type: AttachmentType.IMAGE
};
const result =
	'Generated 1 clip.\n[Attachment saved: mcp-attachment-1-0.wav]\nClip 1: seed 5\n[Attachment saved: mcp-attachment-1-1.png]';

describe('parseToolResultWithMedia', () => {
	it('matches audio and image extras by placeholder name', () => {
		const lines = parseToolResultWithMedia(result, [audio, image]);

		expect(lines.map((line) => line.media?.name)).toEqual([
			undefined,
			audio.name,
			undefined,
			image.name
		]);
	});

	it('recomputes when only audio extras change', () => {
		const first = parseToolResultWithMedia('x\n[Attachment saved: a.wav]', []);
		const second = parseToolResultWithMedia('x\n[Attachment saved: a.wav]', [
			{ ...audio, name: 'a.wav' }
		]);

		expect(first[1].media).toBeUndefined();
		expect(second[1].media?.name).toBe('a.wav');
	});
});

describe('groupToolResultLines', () => {
	it('joins text runs and isolates media, dropping placeholder lines', () => {
		const segments = groupToolResultLines(parseToolResultWithMedia(result, [audio, image]));

		expect(segments).toEqual([
			{ kind: ToolResultSegmentKind.TEXT, text: 'Generated 1 clip.' },
			{ kind: ToolResultSegmentKind.MEDIA, media: audio },
			{ kind: ToolResultSegmentKind.TEXT, text: 'Clip 1: seed 5' },
			{ kind: ToolResultSegmentKind.MEDIA, media: image }
		]);
	});

	it('keeps unmatched placeholders as text', () => {
		const segments = groupToolResultLines(
			parseToolResultWithMedia('a\n[Attachment saved: x]\nb', [])
		);

		expect(segments).toEqual([
			{ kind: ToolResultSegmentKind.TEXT, text: 'a\n[Attachment saved: x]\nb' }
		]);
	});
});
