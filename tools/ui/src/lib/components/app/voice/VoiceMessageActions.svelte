<script lang="ts">
	// Per-message replay: speak a committed assistant message with the
	// currently selected voice. Becomes a stop button while it plays.
	import { Square, Volume2 } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import type { DatabaseMessage } from '$lib/types';
	import { voiceController } from '$lib/voice/controller.svelte';

	interface Props {
		message: DatabaseMessage;
	}

	let { message }: Props = $props();

	const replaying = $derived(voiceController.replayingMsgId === message.id);
</script>

{#if voiceController.available && message.content}
	<div class="mt-2 flex h-6 items-center">
		<ActionIcon
			ariaLabel={replaying ? 'Stop speaking' : 'Speak this message'}
			icon={replaying ? Square : Volume2}
			onclick={() => voiceController.toggleReplay(message.id, message.content)}
			tooltip={replaying ? 'Stop' : 'Speak this message'}
		/>
	</div>
{/if}
