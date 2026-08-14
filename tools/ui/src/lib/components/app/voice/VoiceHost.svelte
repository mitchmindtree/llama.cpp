<script lang="ts">
	// App-wide voice singleton host: mounted once in the root layout. Owns the
	// effects (a component context is required for $effect) while the state
	// lives in the controller.
	import { chatStore, conversationsStore } from '$lib/stores';
	import { voiceController } from '$lib/voice/controller.svelte';
	import { onMount } from 'svelte';

	onMount(() => {
		void voiceController.probe();

		return () => {
			if (voiceController.micEnabled) void voiceController.toggleMic();
		};
	});

	// Streamed assistant text -> sentence chunker -> TTS. The streaming map is a
	// SvelteMap whose entry is replaced on every chunk and deleted when the
	// completion ends, so this re-runs per chunk and sees '' at the end.
	$effect(() => {
		const convId = conversationsStore.activeConversation?.id;

		voiceController.onStreamedText(
			convId ? (chatStore.getChatStreaming(convId)?.response ?? '') : ''
		);
	});
</script>
