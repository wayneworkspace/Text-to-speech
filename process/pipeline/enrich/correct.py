"""
CORRECT stage: ask Claude to review the transcript once and fix likely
speech-to-text errors - mainly misheard proper nouns (names, song/book
titles) and domain-specific technical terms Whisper did not recognize.

Why this exists instead of a hand-maintained find/replace dictionary: you
cannot know in advance every word Whisper might mis-transcribe across many
videos. Claude uses its own general knowledge plus the video's filename
(and DOMAIN_VOCABULARY, if set) as context to catch errors a fixed
dictionary would miss - at the cost of being non-deterministic and unable
to fix a term neither Claude nor Whisper has ever seen. See CODE_REVIEW.md
for the fuller design discussion.

This stage is optional. If ANTHROPIC_API_KEY is not set in .env, main.py
skips calling this module entirely and the transcript is written exactly
as Whisper produced it - the rest of the pipeline still works normally.

Design note: like pipeline/enrich/enrich.py's segment_topics(), Claude is
asked to return only the *lines that need fixing* (index + corrected
text), never the whole transcript back - this keeps the response (and the
API cost) small regardless of transcript length, and is why "did the
response include a line for every segment" is not the safety check here;
instead, every returned index is bounds-checked before anything is
touched.

The transcript is also sent in batches of CONFIG["correction_batch_size"]
lines per call (see correct_transcript_errors()'s `batch_size` arg), not
as one call for the whole video. This was not a design choice up front -
it was added after a real failure running this project on a ~2 hour
video: one call covering all ~1500 lines spent its entire output budget
on the model's own extended thinking and returned no text block at all,
so the JSON parse failed on an empty string and correction silently never
applied for that whole run (see CODE_REVIEW.md). Batching keeps each
call's input/thinking/output small regardless of total video length, the
same way pipeline/transform/transform.py already chunks Whisper
transcription itself. A failed batch only loses that batch's lines - the
rest of the video's corrections still apply.
"""
import json
import os

from ..utils import extract_text_from_anthropic_response, log_anthropic_usage


def _build_numbered_transcript(segments: list, start_index: int = 0) -> str:
    """
    `start_index` lets a batch's line numbers reflect its position in the
    *full* transcript (not 0-based within the batch), so a correction's
    "index" from any batch maps directly onto the original segments list
    with no remapping needed at merge time.
    """
    return "\n".join(f"[{start_index + i}] {seg['text'].strip()}" for i, seg in enumerate(segments))


def _parse_corrections(raw: str) -> dict:
    """Returns {index: corrected_text}. Raises (ValueError/json.JSONDecodeError/
    TypeError/KeyError) on anything unparseable - the caller always catches
    this and falls back to the original, unmodified segments."""
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip a ```json ... ``` fence if Claude added one despite being asked not to.
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    items = json.loads(raw)
    return {
        item["index"]: item["text"]
        for item in items
        if isinstance(item, dict) and "index" in item and "text" in item
    }


def _build_correction_prompt(context_lines: list, transcript: str) -> str:
    return (
        "This is a numbered speech-to-text transcript that may contain mis-transcribed "
        "words - especially proper nouns (names, song/book/place titles) and technical or "
        "domain-specific terms the speech-to-text model did not recognize.\n\n"
        + "\n".join(context_lines) + "\n\n"
        "Find lines with a likely transcription error and propose a corrected version of "
        "just that line's text. Rules:\n"
        "- Only fix clear transcription errors (wrong word, wrong proper noun, garbled "
        "technical term). Do NOT rephrase, summarize, translate, or change meaning.\n"
        "- Keep the same language as the original line.\n"
        "- If a line has no error, leave it out of your reply entirely - do not include it.\n\n"
        'Reply with ONLY a JSON array (no other text, no markdown fence), where each item '
        'looks like {"index": <line number>, "text": <corrected line text>}. Reply with an '
        "empty array [] if nothing needs fixing.\n\n"
        "Transcript:\n" + transcript
    )


def correct_transcript_errors(
    segments: list, video_path: str, domain_vocabulary, api_key: str, model: str, log,
    batch_size: int = 150,
) -> list:
    """
    Ask Claude to fix likely mis-transcribed words, using the video's
    filename and optional DOMAIN_VOCABULARY as context (both free - no need
    to enumerate every term you expect in advance).

    Only the "text" field of a segment is ever changed - start/end/speaker/
    low_confidence are always left untouched, and segments are never added,
    removed, or reordered.

    Sends `segments` to Claude in batches of up to `batch_size` lines (see
    CONFIG["correction_batch_size"] / module docstring for why) instead of
    one call for the whole transcript. A batch that fails or comes back
    malformed only costs that batch's lines - every other batch's
    corrections still apply, so one bad chunk of a long video never
    cancels correction for the whole thing.

    Never raises: a bad key, network error, timeout, or malformed response
    is logged and those lines are left unchanged, so a correction problem
    can never break transcription or desync timestamps with the rest of
    the pipeline.
    """
    if not segments:
        return segments

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    context_lines = [f"Video file name: {os.path.basename(video_path)}"]
    if domain_vocabulary:
        context_lines.append(f"Known domain vocabulary/terms that may appear: {domain_vocabulary}")

    num_batches = (len(segments) + batch_size - 1) // batch_size
    if num_batches > 1:
        log(
            f"Calling Claude to check for likely transcription errors "
            f"({num_batches} batches of up to {batch_size} lines each)..."
        )
    else:
        log("Calling Claude to check for likely transcription errors...")

    all_corrections = {}
    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start: batch_start + batch_size]
        transcript = _build_numbered_transcript(batch, start_index=batch_start)
        prompt = _build_correction_prompt(context_lines, transcript)
        batch_label = (
            f"lines {batch_start}-{batch_start + len(batch) - 1}" if num_batches > 1 else "transcript"
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            log_anthropic_usage(response, log)
            batch_corrections = _parse_corrections(extract_text_from_anthropic_response(response))
        except Exception as exc:
            log(f"Transcript correction skipped for {batch_label} (Claude call failed: {exc})")
            continue

        # Bounds-check every index before touching anything - one bad index
        # from a malformed response must not leave this batch half-trusted.
        # Scoped to this batch only, so it never costs other batches' results.
        if any(idx < 0 or idx >= len(segments) for idx in batch_corrections):
            log(f"Ignored corrections for {batch_label} (Claude returned an out-of-range segment index).")
            continue

        all_corrections.update(batch_corrections)

    if not all_corrections:
        log("No likely transcription errors found.")
        return segments

    changed = 0
    for idx, corrected_text in all_corrections.items():
        if segments[idx]["text"].strip() != corrected_text.strip():
            segments[idx]["text"] = corrected_text
            changed += 1

    log(f"Corrected {changed} likely transcription error(s).")
    return segments
