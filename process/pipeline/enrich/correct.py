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
"""
import json
import os


def _build_numbered_transcript(segments: list) -> str:
    return "\n".join(f"[{i}] {seg['text'].strip()}" for i, seg in enumerate(segments))


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


def correct_transcript_errors(
    segments: list, video_path: str, domain_vocabulary, api_key: str, model: str, log
) -> list:
    """
    Ask Claude to fix likely mis-transcribed words, using the video's
    filename and optional DOMAIN_VOCABULARY as context (both free - no need
    to enumerate every term you expect in advance).

    Only the "text" field of a segment is ever changed - start/end/speaker/
    low_confidence are always left untouched, and segments are never added,
    removed, or reordered.

    Never raises: a bad key, network error, timeout, or malformed response
    is logged and the original segments are returned unchanged, so a
    correction problem can never break transcription or desync timestamps
    with the rest of the pipeline.
    """
    if not segments:
        return segments

    log("Calling Claude to check for likely transcription errors...")
    import anthropic

    context_lines = [f"Video file name: {os.path.basename(video_path)}"]
    if domain_vocabulary:
        context_lines.append(f"Known domain vocabulary/terms that may appear: {domain_vocabulary}")

    transcript = _build_numbered_transcript(segments)
    prompt = (
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

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        corrections = _parse_corrections(response.content[0].text)
    except Exception as exc:
        log(f"Transcript correction skipped (Claude call failed: {exc})")
        return segments

    if not corrections:
        log("No likely transcription errors found.")
        return segments

    # Bounds-check every index before touching anything - one bad index
    # from a malformed response must not leave the transcript half-edited.
    if any(idx < 0 or idx >= len(segments) for idx in corrections):
        log("Transcript correction skipped (Claude returned an out-of-range segment index).")
        return segments

    changed = 0
    for idx, corrected_text in corrections.items():
        if segments[idx]["text"].strip() != corrected_text.strip():
            segments[idx]["text"] = corrected_text
            changed += 1

    log(f"Corrected {changed} likely transcription error(s).")
    return segments
