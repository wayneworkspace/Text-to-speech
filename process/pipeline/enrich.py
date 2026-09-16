"""
ENRICH stage: ask Claude to read the flattened transcript once and mark
where each new topic starts.

This stage is optional. If ANTHROPIC_API_KEY is not set in .env, main.py
skips calling this module entirely and the transcript is written as one
flat list with no topic headings - the rest of the pipeline still works
normally.

Design note: Claude is asked to return only the *positions* of topic
changes (a segment index + a short title), never to repeat the transcript
back - this keeps the response (and the API cost) small no matter how long
the source transcript is.
"""
import json


def _build_numbered_transcript(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments):
        speaker = f"{seg['speaker']}: " if seg.get("speaker") else ""
        lines.append(f"[{i}] {speaker}{seg['text'].strip()}")
    return "\n".join(lines)


def _parse_topics(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip a ```json ... ``` fence if Claude added one despite being asked not to.
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    topics = json.loads(raw)
    return [t for t in topics if isinstance(t, dict) and "index" in t and "title" in t]


def segment_topics(segments: list, api_key: str, model: str, log) -> list:
    """
    Call Claude once on the whole transcript. Returns a list of
    {"index": int, "title": str} marking where each topic starts.

    Never raises - any failure (bad key, network error, malformed response)
    is logged and an empty list is returned, so a topic-segmentation problem
    never breaks transcription or the rest of the pipeline.
    """
    if not segments:
        return []

    log("Calling Claude to detect topic changes...")
    import anthropic

    transcript = _build_numbered_transcript(segments)
    prompt = (
        "This is a numbered, timestamped transcript of a meeting or conversation. "
        "Read it and identify where the discussion moves to a genuinely new topic.\n\n"
        "Reply with ONLY a JSON array (no other text, no markdown fence), where each "
        'item looks like {"index": <segment number where the new topic starts>, '
        '"title": <short topic title>}. Always include index 0. Keep the number of '
        "topics reasonable - do not start a new topic for every sentence.\n\n"
        "Transcript:\n" + transcript
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        topics = _parse_topics(response.content[0].text)
        log(f"Found {len(topics)} topic(s).")
        return topics
    except Exception as exc:
        log(f"Topic segmentation skipped (Claude call failed: {exc})")
        return []
