<script lang="ts">
	// App-wide voice singleton host: mounted once in the root layout. Owns the
	// effects (a component context is required for $effect) while the state
	// lives in the controller.
	import { onMount } from 'svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { voiceController } from '$lib/voice/controller.svelte';

	onMount(() => {
		void voiceController.probe();
		return () => {
			if (voiceController.micEnabled) void voiceController.toggleMic();
		};
	});

	// Streamed assistant text -> sentence chunker -> TTS.
	$effect(() => {
		voiceController.onStreamedText(chatStore.currentResponse);
	});
</script>
