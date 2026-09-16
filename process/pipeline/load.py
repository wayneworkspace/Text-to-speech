"""
LOAD stage: write the final result (list of segments) to storage.
Currently the only destination is a Markdown file, but this is kept as its
own module so other destinations (e.g. .srt, .txt, a database...) can be
added later without touching EXTRACT/TRANSFORM.
"""
import os
from datetime import datetime

from .utils import format_timestamp


def build_output_filename(video_path: str) -> str:
    """Naming convention: [dd-mm-yy] - <video_name> - transcript.md (date = the run date)."""
    date_str = datetime.now().strftime("%d-%m-%y")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"[{date_str}] - {video_name} - transcript.md"


def write_markdown(output_dir: str, video_path: str, segments: list, topics: list = None) -> str:
    """
    Write the segment list to a Markdown file inside output_dir, return the
    file path.

    `topics` is an optional list of {"index": int, "title": str} (see
    pipeline/enrich.py) - a "## <title>" heading is inserted right before the
    segment at that index. Segments carrying a "speaker" label (see
    pipeline/diarize.py) are prefixed with it, and any segment flagged
    "low_confidence" (see pipeline/transform.py) gets a warning marker so the
    reader knows which lines are worth double-checking.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = build_output_filename(video_path)
    output_path = os.path.join(output_dir, filename)

    topic_at_index = {t["index"]: t["title"] for t in (topics or [])}

    lines = [f"# Transcript: {os.path.basename(video_path)}", ""]
    if any(seg.get("low_confidence") for seg in segments):
        lines.append("> ⚠️ marks a line Whisper was not confident about - worth a manual check.")
        lines.append("")

    for i, seg in enumerate(segments):
        if i in topic_at_index:
            lines.append(f"## {topic_at_index[i]}")
            lines.append("")

        text = seg["text"].strip()
        if not text:
            continue

        ts = format_timestamp(seg["start"])
        prefix = f"**[{ts}] {seg['speaker']}:**" if seg.get("speaker") else f"**[{ts}]**"
        flag = " ⚠️" if seg.get("low_confidence") else ""
        lines.append(f"{prefix} {text}{flag}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
