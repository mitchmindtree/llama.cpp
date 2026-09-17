import { AgenticSectionType, AttachmentType } from '$lib/enums';
import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
import { extractPromptArg, extractToolResultMedia } from '$lib/utils/agentic';
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
const section = (toolResult: string, extras = [audio, image]): AgenticSection => ({
	content: toolResult,
	toolResult,
	toolResultExtras: extras,
	type: AgenticSectionType.TOOL_CALL
});

describe('extractToolResultMedia', () => {
	it('pairs each attachment with the text line that follows it', () => {
		const items = extractToolResultMedia(
			section(
				'Generated 2 clips.\n[Attachment saved: mcp-attachment-1-0.wav]\nClip 1: seed 5\n[Attachment saved: mcp-attachment-1-1.png]'
			)
		);

		expect(items).toEqual([{ caption: 'Clip 1: seed 5', media: audio }, { media: image }]);
	});

	it('gives no caption when the next line is another attachment', () => {
		const items = extractToolResultMedia(
			section(
				'[Attachment saved: mcp-attachment-1-0.wav]\n[Attachment saved: mcp-attachment-1-1.png]\ntail'
			)
		);

		expect(items).toEqual([{ media: audio }, { caption: 'tail', media: image }]);
	});

	it('returns nothing without extras or without a result', () => {
		expect(extractToolResultMedia(section('[Attachment saved: x]', []))).toEqual([]);
		expect(extractToolResultMedia({ content: '', type: AgenticSectionType.TOOL_CALL })).toEqual([]);
	});
});

describe('extractPromptArg', () => {
	it('returns the trimmed prompt string', () => {
		expect(extractPromptArg('{"prompt":"  door creak ","count":2}')).toBe('door creak');
	});

	it('ignores missing, empty, non-string or unparsable prompts', () => {
		expect(extractPromptArg('{"query":"x"}')).toBeUndefined();
		expect(extractPromptArg('{"prompt":"  "}')).toBeUndefined();
		expect(extractPromptArg('{"prompt":3}')).toBeUndefined();
		expect(extractPromptArg('{"prompt":')).toBeUndefined();
		expect(extractPromptArg(undefined)).toBeUndefined();
	});
});
