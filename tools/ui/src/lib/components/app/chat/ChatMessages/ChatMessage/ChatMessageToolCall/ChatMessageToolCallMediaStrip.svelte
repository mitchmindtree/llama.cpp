<script lang="ts">
	// Players and images of a tool result shown in the message body while the
	// tool block is collapsed, so media is reachable without expanding it. The
	// prompt argument heads the strip and each item carries the text line that
	// followed it in the result (a clip's seed line, for example).

	import ChatMessageToolCallMedia from './ChatMessageToolCallMedia.svelte';
	import type { AgenticSection } from '$lib/types';
	import { extractPromptArg, extractToolResultMedia } from '$lib/utils';

	interface Props {
		section: AgenticSection;
	}

	let { section }: Props = $props();

	const items = $derived(extractToolResultMedia(section));
	const prompt = $derived(extractPromptArg(section.toolArgs));
</script>

{#if items.length > 0}
	<div class="tool-media-strip grid gap-2">
		{#if prompt}
			<div class="text-sm text-muted-foreground italic">"{prompt}"</div>
		{/if}

		{#each items as item, i (i)}
			<div>
				{#if item.caption}
					<div class="text-xs text-muted-foreground">{item.caption}</div>
				{/if}

				<ChatMessageToolCallMedia class="mt-1" media={item.media} />
			</div>
		{/each}
	</div>
{/if}
