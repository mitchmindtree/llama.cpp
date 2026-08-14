<script lang="ts">
	// Voice management: registered voices (preview, default, delete), cloning
	// a new voice from a recording or file, and designing a voice from a
	// free-text description via the voicedesign model.
	import { Check, CircleStop, Mic, Play, Square, Trash2, Upload } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { SETTINGS_KEYS } from '$lib/constants/settings-keys';
	import { getChatSettingsConfigContext } from '$lib/contexts';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { AudioRecorder, convertToWav } from '$lib/utils/audio-recording';
	import * as gateway from '$lib/voice/gateway';
	import { voiceController } from '$lib/voice/controller.svelte';
	import { onMount } from 'svelte';

	let busy = $state('');
	let error = $state('');

	// The settings page commits its own mount-time snapshot (localConfig) of
	// the WHOLE config on "Save settings". Writing only to the store would get
	// clobbered by that snapshot, so every change is mirrored into it too.
	const settingsCtx = getChatSettingsConfigContext();

	function setVoiceConfig(key: string, value: string | number | boolean) {
		settingsStore.updateConfig(key as never, value as never);
		settingsCtx?.handleConfigChange?.(key, value as string | boolean);
	}

	// Clone section
	let cloneName = $state('');
	let cloneTranscript = $state('');
	let cloneFile = $state<File | null>(null);
	let recorder: AudioRecorder | null = null;
	let recording = $state(false);
	let recordedBlob = $state<Blob | null>(null);
	let lastClonedTranscript = $state('');

	// Design section
	let designInstructions = $state('');
	let designText = $state(
		'Well met, traveler. The road ahead is long, but the stories make it worth walking.'
	);
	let designName = $state('');
	let designBlob = $state<Blob | null>(null);
	let designUrl = $state('');

	onMount(() => {
		void voiceController.refreshVoices();
		return () => {
			if (designUrl) URL.revokeObjectURL(designUrl);
		};
	});

	async function run(label: string, fn: () => Promise<void>) {
		busy = label;
		error = '';
		try {
			await fn();
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			busy = '';
		}
	}

	function setDefault(name: string) {
		setVoiceConfig(SETTINGS_KEYS.VOICE_VOICE, name);
	}

	async function toggleRecord() {
		if (recording) {
			recording = false;
			const blob = await recorder!.stopRecording();
			recordedBlob = await convertToWav(blob);
			cloneFile = null;
		} else {
			recorder = new AudioRecorder();
			await recorder.startRecording();
			recordedBlob = null;
			recording = true;
		}
	}

	async function clone() {
		const source = cloneFile ?? recordedBlob;
		if (!cloneName || !source) return;
		await run('cloning', async () => {
			const form = new FormData();
			form.set('name', cloneName);
			form.set('file', source, 'reference.wav');
			if (cloneTranscript.trim()) form.set('transcript', cloneTranscript.trim());
			const result = await gateway.cloneVoice(form);
			lastClonedTranscript = result.transcript;
			cloneName = '';
			cloneTranscript = '';
			cloneFile = null;
			recordedBlob = null;
			await voiceController.refreshVoices();
		});
	}

	async function remove(name: string) {
		await run(`deleting ${name}`, async () => {
			await gateway.deleteVoice(name);
			await voiceController.refreshVoices();
		});
	}

	async function generateDesign() {
		if (!designInstructions.trim() || !designText.trim()) return;
		await run('designing', async () => {
			designBlob = await gateway.design({
				text: designText.trim(),
				instructions: designInstructions.trim()
			});
			if (designUrl) URL.revokeObjectURL(designUrl);
			designUrl = URL.createObjectURL(designBlob);
		});
	}

	async function saveDesign() {
		if (!designName || !designBlob) return;
		await run('saving voice', async () => {
			const form = new FormData();
			form.set('name', designName);
			form.set('file', designBlob!, 'design.wav');
			form.set('transcript', designText.trim());
			await gateway.cloneVoice(form);
			designName = '';
			await voiceController.refreshVoices();
		});
	}
</script>

{#if !voiceController.available}
	<div class="py-8 text-center text-sm text-muted-foreground">
		Voice gateway unavailable. Voice needs the HTTPS origin (the mic and audio are disabled on
		plain http) and a reachable gateway.
	</div>
{:else}
	<!-- min-w-0/overflow-hidden: the parent is a CSS grid item whose min-width
	     defaults to auto, so nowrap content (truncated transcripts) would
	     otherwise blow the page out sideways. -->
	<div class="min-w-0 space-y-8 overflow-hidden">
		{#if error}
			<div class="rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm">{error}</div>
		{/if}

		<section class="space-y-3">
			<h4 class="text-sm font-semibold">Voices</h4>
			<p class="text-xs text-muted-foreground">
				The default voice speaks all replies. A reply segment starting with [voice:name] switches
				to that voice, which lets a DM persona give NPCs their own voices.
			</p>
			<!-- ~10 rows before scrolling, so cloning and design stay reachable. -->
			<div class="max-h-[26rem] space-y-2 overflow-y-auto pr-1">
				{#each voiceController.voices as voice (voice.name)}
					<div class="flex min-w-0 items-center gap-2 rounded-md border border-border/40 p-2">
						<span class="font-mono text-sm">{voice.name}</span>
						{#if config().voiceVoice === voice.name}
							<span class="rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-400">
								default
							</span>
						{/if}
						<span class="min-w-0 flex-1 truncate text-xs text-muted-foreground">
							{voice.ref_text}
						</span>
						<Button
							class="h-7 w-7 p-0"
							onclick={() => voiceController.togglePreview(voice.name)}
							title={voiceController.previewingVoice === voice.name ? 'Stop' : 'Preview'}
							variant="ghost"
						>
							{#if voiceController.previewingVoice === voice.name}
								<Square class="{ICON_CLASS_DEFAULT} text-red-400" />
							{:else}
								<Play class={ICON_CLASS_DEFAULT} />
							{/if}
						</Button>
						<Button
							class="h-7 w-7 p-0"
							disabled={config().voiceVoice === voice.name}
							onclick={() => setDefault(voice.name)}
							title="Use as default voice"
							variant="ghost"
						>
							<Check class={ICON_CLASS_DEFAULT} />
						</Button>
						<Button
							class="h-7 w-7 p-0 text-red-400"
							disabled={busy !== ''}
							onclick={() => remove(voice.name)}
							title="Delete"
							variant="ghost"
						>
							<Trash2 class={ICON_CLASS_DEFAULT} />
						</Button>
					</div>
				{:else}
					<div class="text-sm text-muted-foreground">No voices registered yet.</div>
				{/each}
			</div>
		</section>

		<section class="space-y-3">
			<h4 class="text-sm font-semibold">Playback</h4>
			<label class="flex flex-col gap-1 text-sm">
				<span>
					Expressiveness (TTS temperature):
					<span class="font-mono">{Number(config().voiceTemperature ?? 0.9).toFixed(2)}</span>
				</span>
				<input
					class="w-64"
					max="1.3"
					min="0.3"
					onchange={(e) => setVoiceConfig('voiceTemperature', Number(e.currentTarget.value))}
					step="0.05"
					type="range"
					value={Number(config().voiceTemperature ?? 0.9)}
				/>
				<span class="text-xs text-muted-foreground">
					Lower is flatter and more stable, higher is livelier with more variation between
					takes. 0.90 is the model default.
				</span>
			</label>
			<label class="flex items-center gap-2 text-sm">
				<Checkbox
					checked={config().voiceParagraphChunks !== false}
					onCheckedChange={(v) => setVoiceConfig('voiceParagraphChunks', v === true)}
				/>
				<span>
					Paragraph-scale speech chunks: synthesize several sentences per request for smoother,
					more connected prosody. Off = one sentence per request (earlier behavior, faster
					reaction to very short replies). Applies from the next reply.
				</span>
			</label>
			<label class="flex items-center gap-2 text-sm">
				<Checkbox
					checked={Boolean(config().voiceHalfDuplex)}
					onCheckedChange={(v) => setVoiceConfig('voiceHalfDuplex', v === true)}
				/>
				<span>
					Half-duplex (for speakers): mute mic input while the assistant is speaking. Use when
					echo cancellation cannot stop the assistant from hearing itself. Disables voice
					barge-in.
				</span>
			</label>
		</section>

		<section class="space-y-3">
			<h4 class="text-sm font-semibold">Clone a voice</h4>
			<p class="text-xs text-muted-foreground">
				Record or upload 10-15 seconds of clean speech. The clip's mood carries into everything
				the voice says, so record in the delivery you want (a calm read gives a calm voice, an
				excited read an excited one). Without a transcript, the clip is transcribed
				automatically (shown below after cloning; delete and redo if it is wrong). Voices cloned
				before the latest improvements can be re-cloned to pick them up.
			</p>
			<div class="flex flex-wrap items-center gap-2">
				<Input bind:value={cloneName} class="w-40" placeholder="voice name" />
				<Button disabled={busy !== ''} onclick={toggleRecord} type="button" variant="outline">
					{#if recording}
						<CircleStop class="{ICON_CLASS_DEFAULT} animate-pulse text-red-400" />
						Stop
					{:else}
						<Mic class={ICON_CLASS_DEFAULT} />
						Record
					{/if}
				</Button>
				<label class="inline-flex cursor-pointer items-center gap-1 text-sm">
					<Upload class={ICON_CLASS_DEFAULT} />
					<span>{cloneFile ? cloneFile.name : 'or upload a clip'}</span>
					<input
						accept="audio/*"
						class="hidden"
						onchange={(e) => {
							cloneFile = (e.currentTarget as HTMLInputElement).files?.[0] ?? null;
							recordedBlob = null;
						}}
						type="file"
					/>
				</label>
				{#if recordedBlob}
					<span class="text-xs text-green-400">clip recorded</span>
				{/if}
				<Button
					disabled={busy !== '' || !cloneName || (!cloneFile && !recordedBlob)}
					onclick={clone}
					type="button"
				>
					{busy === 'cloning' ? 'Cloning...' : 'Clone'}
				</Button>
			</div>
			<Input bind:value={cloneTranscript} placeholder="transcript (optional, auto-transcribed if empty)" />
			{#if lastClonedTranscript}
				<p class="text-xs text-muted-foreground">Transcribed as: "{lastClonedTranscript}"</p>
			{/if}
		</section>

		<section class="space-y-3">
			<h4 class="text-sm font-semibold">Design a voice</h4>
			<p class="text-xs text-muted-foreground">
				Describe a voice (age, tone, accent, character), audition it on the sample text, then
				save it as a reusable voice.
			</p>
			<Textarea
				bind:value={designInstructions}
				placeholder="e.g. old gravelly dwarven blacksmith, slow, gruff, deep voice"
				rows={2}
			/>
			<Input bind:value={designText} placeholder="sample text to speak" />
			<div class="flex flex-wrap items-center gap-2">
				<Button
					disabled={busy !== '' || !designInstructions.trim() || !designText.trim()}
					onclick={generateDesign}
					type="button"
					variant="outline"
				>
					{busy === 'designing' ? 'Generating...' : 'Generate'}
				</Button>
				{#if designUrl}
					<audio class="max-w-full" controls src={designUrl}></audio>
					<Input bind:value={designName} class="w-40" placeholder="save as name" />
					<Button disabled={busy !== '' || !designName} onclick={saveDesign} type="button">
						{busy === 'saving voice' ? 'Saving...' : 'Save as voice'}
					</Button>
				{/if}
			</div>
		</section>
	</div>
{/if}
