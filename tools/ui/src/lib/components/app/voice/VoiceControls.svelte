<script lang="ts">
	// Compact voice controls for the chat form: speak-replies toggle and the
	// continuous-conversation mic toggle. Hidden entirely when the gateway is
	// unreachable or the context is insecure (plain http cannot open the mic).
	import { AudioLines, Mic, Volume2, VolumeX } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { voiceController } from '$lib/voice/controller.svelte';

	const statusColor = {
		idle: 'bg-muted-foreground/40',
		listening: 'bg-green-500',
		speaking: 'bg-blue-500 animate-pulse',
		transcribing: 'bg-yellow-500 animate-pulse'
	} as const;
</script>

{#if voiceController.available}
	<div class="flex items-center gap-1">
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					class="h-8 w-8 rounded-full p-0 {voiceController.speakEnabled
						? 'text-blue-500'
						: 'text-muted-foreground'}"
					onclick={() => voiceController.toggleSpeak()}
					type="button"
					variant="ghost"
				>
					<span class="sr-only">Toggle spoken replies</span>

					{#if voiceController.speakEnabled}
						<Volume2 class={ICON_CLASS_DEFAULT} />
					{:else}
						<VolumeX class={ICON_CLASS_DEFAULT} />
					{/if}
				</Button>
			</Tooltip.Trigger>

			<Tooltip.Content>Speak replies aloud</Tooltip.Content>
		</Tooltip.Root>

		<Tooltip.Root>
			<Tooltip.Trigger>
				<span class="relative inline-flex">
					{#if voiceController.micEnabled}
						<!-- Input-level ring: breathes with detected speech. -->
						<span
							class="pointer-events-none absolute inset-0 rounded-full border-2 border-green-500 transition-[opacity,transform] duration-100"
							style="opacity: {voiceController.micLevel}; transform: scale({1 +
								voiceController.micLevel * 0.25});"
						></span>
					{/if}

					<Button
						class="h-8 w-8 rounded-full p-0 {voiceController.micEnabled
							? 'text-green-500'
							: 'text-muted-foreground'}"
						onclick={() => voiceController.toggleMic()}
						type="button"
						variant="ghost"
					>
						<span class="sr-only">Toggle voice conversation</span>

						{#if voiceController.micEnabled}
							<AudioLines class={ICON_CLASS_DEFAULT} />
						{:else}
							<Mic class={ICON_CLASS_DEFAULT} />
						{/if}
					</Button>
				</span>
			</Tooltip.Trigger>

			<Tooltip.Content>Continuous voice conversation</Tooltip.Content>
		</Tooltip.Root>

		{#if voiceController.speakEnabled || voiceController.micEnabled}
			<span class="h-2 w-2 rounded-full {statusColor[voiceController.status]}"></span>

			{#if voiceController.status !== 'idle'}
				<span class="hidden text-xs text-muted-foreground select-none sm:inline">
					{voiceController.status === 'transcribing' ? 'transcribing...' : voiceController.status}
				</span>
			{/if}
		{/if}
	</div>
{/if}
