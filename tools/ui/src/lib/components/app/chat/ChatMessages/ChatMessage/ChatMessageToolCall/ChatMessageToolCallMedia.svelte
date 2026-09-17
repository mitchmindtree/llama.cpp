<script lang="ts">
	// Inline player or image for one tool-result attachment, with a link that
	// downloads the original bytes under the attachment's name. Shared by the
	// tool call blocks so every media result gets the same treatment.

	import { Download } from '@lucide/svelte';
	import { AttachmentType, MimeTypeAudio } from '$lib/enums';
	import type { DatabaseMessageExtraAudioFile, DatabaseMessageExtraImageFile } from '$lib/types';
	import { createBase64DataUrl } from '$lib/utils/data-url';

	interface Props {
		media: DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile;
		alt?: string;
		class?: string;
		imageClass?: string;
	}

	let {
		alt,
		class: className = 'mt-2 mb-2',
		imageClass = 'h-auto max-w-full rounded-lg',
		media
	}: Props = $props();

	const audioMimeType = $derived(
		media.type === AttachmentType.AUDIO ? (media.mimeType ?? MimeTypeAudio.MP3_MPEG) : undefined
	);
	// One data URL serves both the element and the download link, so the
	// multi-megabyte string exists once per attachment.
	const src = $derived(
		media.type === AttachmentType.AUDIO
			? createBase64DataUrl(audioMimeType ?? MimeTypeAudio.MP3_MPEG, media.base64Data)
			: media.base64Url
	);
</script>

<div class={className}>
	{#if media.type === AttachmentType.AUDIO}
		<audio class="w-full rounded-lg" controls>
			<source {src} type={audioMimeType} />
			Your browser does not support the audio element.
		</audio>
	{:else}
		<img alt={alt ?? media.name} class={imageClass} loading="lazy" {src} />
	{/if}

	<a
		class="mt-1 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
		download={media.name}
		href={src}
	>
		<Download class="h-3 w-3" />
		{media.name}
	</a>
</div>
