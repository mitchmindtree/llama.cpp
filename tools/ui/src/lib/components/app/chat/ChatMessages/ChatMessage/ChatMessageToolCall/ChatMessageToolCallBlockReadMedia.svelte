<script lang="ts">
	import ChatMessageToolCallMedia from './ChatMessageToolCallMedia.svelte';
	import { parseReadMediaMeta } from './parsers/read-media';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { ATTACHMENT_SAVED_REGEX } from '$lib/constants/agentic.constants';
	import { AttachmentType } from '$lib/enums';
	import type { DatabaseMessageExtraAudioFile, DatabaseMessageExtraImageFile } from '$lib/types';
	import type { AgenticSection } from '$lib/types';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const readMediaMeta = $derived(parseReadMediaMeta(section));

	// extractBase64Attachments swapped the data URI line for [Attachment saved: name]
	// and moved the bytes to the message extras, so the name is the only link back
	const mediaAttachment = $derived.by(() => {
		const extras = section.toolResultExtras;

		if (!extras || extras.length === 0) return null;

		const match = section.toolResult?.match(ATTACHMENT_SAVED_REGEX);

		if (!match) return null;

		const attachmentName = match[1];

		return (
			extras.find(
				(e): e is DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile =>
					(e.type === AttachmentType.IMAGE || e.type === AttachmentType.AUDIO) &&
					e.name === attachmentName
			) ?? null
		);
	});
</script>

<ToolCallBlock {isStreaming} meta={readMediaMeta} {onToggle} {open} {section}>
	{#snippet titleSnippet()}
		<span class="text-muted-foreground">Read media </span>

		<span class="font-mono">{readMediaMeta?.fileName}</span>
	{/snippet}

	{#snippet children(_meta, _ctx)}
		{#if section.toolResult}
			{#if !mediaAttachment}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Media attachment not found in message extras
				</div>
			{:else}
				<ChatMessageToolCallMedia
					alt={readMediaMeta?.fileName ?? 'media'}
					class="mt-2"
					imageClass="max-h-[60vh] max-w-full rounded-lg object-contain shadow-lg"
					media={mediaAttachment}
				/>
			{/if}

			{#if readMediaMeta?.sizeBytes || readMediaMeta?.mimeType}
				<div class="mt-2 flex gap-4 text-xs text-muted-foreground">
					{#if readMediaMeta?.sizeBytes}
						<span>Size: {readMediaMeta.sizeBytes} bytes</span>
					{/if}

					{#if readMediaMeta?.mimeType}
						<span>MIME: {readMediaMeta.mimeType}</span>
					{/if}
				</div>
			{/if}

			{#if readMediaMeta?.path}
				<div class="mt-1 font-mono text-xs text-muted-foreground/60">{readMediaMeta.path}</div>
			{/if}
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for media data...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
