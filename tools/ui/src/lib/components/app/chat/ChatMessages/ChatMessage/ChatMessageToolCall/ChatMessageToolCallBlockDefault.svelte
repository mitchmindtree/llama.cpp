<script lang="ts">
	// Fall-through renderer for tool calls without a dedicated block.
	// Renders section.toolArgs / section.toolResult directly using the
	// shared chrome shell.

	import ChatMessageToolCallMedia from './ChatMessageToolCallMedia.svelte';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { Loader2 } from '@lucide/svelte';
	import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
	import { MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import { FileTypeText, ToolResultKind, ToolResultSegmentKind } from '$lib/enums';
	import type {
		AgenticSection,
		DatabaseMessageExtra,
		ToolResultLine,
		ToolResultSegment
	} from '$lib/types';
	import {
		classifyToolResult,
		formatJsonPretty,
		getToolUi,
		groupToolResultLines,
		parseToolResultWithMedia
	} from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { attachments, isStreaming, onToggle, open, section }: Props = $props();

	const title = $derived(getToolUi(section.toolName)?.label ?? section.toolName ?? '');
	const outputKind = $derived(classifyToolResult(section.toolResult));
	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithMedia(section.toolResult, attachments) : []
	);
	// Markdown results render each text run as markdown and each attachment as
	// a player or image in between, so media survives the markdown branch.
	const segments: ToolResultSegment[] = $derived(groupToolResultLines(parsedLines));
</script>

<ToolCallBlock {isStreaming} meta={null} {onToggle} {open} {section} {title}>
	{#snippet children(_meta, ctx)}
		{#if ctx.isStreamingCall}
			<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
				<span>Input</span>

				{#if ctx.isStreaming}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>

			{#if section.toolArgs}
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{:else if ctx.isStreaming}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Receiving arguments...
				</div>
			{:else}
				<div
					class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
				>
					Response was truncated
				</div>
			{/if}
		{:else}
			{@const showInput = Boolean(section.toolArgs)}
			{#if showInput}
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					<span>Input</span>
				</div>

				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs ?? '')}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{/if}

			<div
				class={showInput
					? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
					: 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
			>
				<span>Output</span>

				{#if ctx.isPending}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>

			{#if ctx.isPending}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Waiting for result...
				</div>
			{:else if section.toolResult}
				{#if outputKind === ToolResultKind.JSON}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolResult)}
						language={FileTypeText.JSON}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
					/>
				{:else if outputKind === ToolResultKind.MARKDOWN}
					{#each segments as segment, i (i)}
						{#if segment.kind === ToolResultSegmentKind.MEDIA}
							<ChatMessageToolCallMedia media={segment.media} />
						{:else}
							<MarkdownContent {attachments} content={segment.text} />
						{/if}
					{/each}
				{:else}
					<div class="overflow-auto">
						{#each parsedLines as line, i (i)}
							<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
								{line.text}
							</div>

							{#if line.media}
								<ChatMessageToolCallMedia media={line.media} />
							{/if}
						{/each}
					</div>
				{/if}
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
			{/if}
		{/if}
	{/snippet}
</ToolCallBlock>
