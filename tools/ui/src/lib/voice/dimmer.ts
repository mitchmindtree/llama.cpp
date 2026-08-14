/**
 * Spoken-progress dimming: while the assistant speaks, the not-yet-spoken
 * remainder of the streaming message renders dimmed via the CSS Custom
 * Highlight API (::highlight(voice-unspoken) rule in app.css, added by the
 * wiring patch). No DOM mutation, no renderer fork.
 *
 * Mapping: the message DOM renders message.content verbatim, and rendered
 * blocks carry data-block-id="hast-<start>-<end>" source offsets, so a raw
 * character boundary (from the chunker's segment offsets) selects the
 * straddling block directly. Blocks past the boundary dim whole; within the
 * straddling block the start lands at the proportional visible-character
 * position snapped to a word boundary. When the content contains LaTeX the
 * offsets shift (preprocessLaTeX), so fall back to whole-block granularity.
 *
 * Timing: the markdown renderer coalesces re-renders to animation frames and
 * replaces its unstable tail block wholesale, so ranges computed just before
 * a swap die on detached nodes. All triggers therefore go through a single
 * rAF-coalesced scheduler, and a MutationObserver on the content element
 * re-schedules immediately after every renderer swap, which is what keeps the
 * highlight from flickering.
 */

const HIGHLIGHT_NAME = 'voice-unspoken';
const BLOCK_ID = /^hast-(\d+)-(\d+)$/;
// Subtrees whose textContent duplicates or does not correspond to source
// text; dimmed atomically rather than walked.
const ATOMIC_SELECTOR =
	'.katex, pre, .code-block-wrapper, .mermaid-block-wrapper, .svg-block-wrapper';

export interface DimState {
	boundary: number;
	blockAligned: boolean;
}

/** Returns the current dim target, or null when the highlight should clear. */
export type DimProvider = () => DimState | null;

let raf: number | null = null;
let provider: DimProvider | null = null;
let observer: MutationObserver | null = null;
let observed: Element | null = null;

function supported(): boolean {
	return typeof CSS !== 'undefined' && 'highlights' in CSS;
}

export function scheduleDim(nextProvider: DimProvider): void {
	if (!supported()) return;
	provider = nextProvider;
	raf ??= requestAnimationFrame(run);
}

export function clearDim(): void {
	provider = null;
	if (raf !== null) {
		cancelAnimationFrame(raf);
		raf = null;
	}
	observer?.disconnect();
	observer = null;
	observed = null;
	if (supported()) CSS.highlights.delete(HIGHLIGHT_NAME);
}

function run(): void {
	raf = null;
	const state = provider?.() ?? null;
	if (state === null) {
		clearDim();
		return;
	}
	const content = contentElement();
	if (!content || !content.isConnected) {
		CSS.highlights.delete(HIGHLIGHT_NAME);
		return;
	}
	observe(content);
	draw(content, state);
}

function contentElement(): Element | null {
	const els = document.querySelectorAll(
		'.chat-message:last-child .chat-message-assistant .agentic-text .markdown-content'
	);
	return els.length > 0 ? els[els.length - 1] : null;
}

/** Re-schedule after every renderer mutation (tail-block swaps). */
function observe(content: Element): void {
	if (observed === content && observed.isConnected) return;
	observer?.disconnect();
	observer = new MutationObserver(() => {
		if (provider !== null) raf ??= requestAnimationFrame(run);
	});
	observer.observe(content, { childList: true, subtree: true, characterData: true });
	observed = content;
}

/** Visible-character position within a block, skipping atomic subtrees. */
function positionInBlock(
	block: Element,
	targetChars: number
): { node: Node; offset: number } | null {
	const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
		acceptNode: (node) =>
			(node.parentElement?.closest(ATOMIC_SELECTOR) ?? null) === null
				? NodeFilter.FILTER_ACCEPT
				: NodeFilter.FILTER_REJECT
	});
	let seen = 0;
	for (let node = walker.nextNode(); node; node = walker.nextNode()) {
		const len = node.textContent?.length ?? 0;
		if (seen + len >= targetChars) {
			let offset = targetChars - seen;
			// Snap back to a word boundary so the dim edge never splits a word.
			const text = node.textContent ?? '';
			while (offset > 0 && offset < text.length && !/\s/.test(text[offset - 1])) offset -= 1;
			return { node, offset };
		}
		seen += len;
	}
	return null;
}

function draw(content: Element, { boundary, blockAligned }: DimState): void {
	const ranges: Range[] = [];
	for (const block of content.querySelectorAll('.markdown-block')) {
		if (!block.isConnected) continue;
		const m = BLOCK_ID.exec(block.getAttribute('data-block-id') ?? '');
		// The unstable tail block has no offsets; being the newest text it is
		// always past the spoken boundary, so it dims whole.
		const start = m ? Number(m[1]) : Number.POSITIVE_INFINITY;
		const end = m ? Number(m[2]) : Number.POSITIVE_INFINITY;

		try {
			if (start >= boundary) {
				const range = new Range();
				range.selectNodeContents(block);
				ranges.push(range);
			} else if (end > boundary && !blockAligned) {
				const visible = positionInBlock(
					block,
					Math.floor(
						((boundary - start) / Math.max(1, end - start)) * (block.textContent?.length ?? 0)
					)
				);
				if (visible) {
					const range = new Range();
					range.setStart(visible.node, visible.offset);
					range.setEndAfter(block);
					ranges.push(range);
				}
			} else if (end > boundary) {
				const range = new Range();
				range.selectNodeContents(block);
				ranges.push(range);
			}
		} catch {
			// Detached or mutated mid-walk: skip, the next tick redraws.
		}
	}

	if (ranges.length === 0) {
		CSS.highlights.delete(HIGHLIGHT_NAME);
		return;
	}
	CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(...ranges));
}
