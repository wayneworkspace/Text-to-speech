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


def write_markdown(output_dir: str, video_path: str, segments: list) -> str:
    """Write the segment list to a Markdown file inside output_dir, return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = build_output_filename(video_path)
    output_path = os.path.join(output_dir, filename)

    lines = [f"# Transcript: {os.path.basename(video_path)}", ""]
    for seg in segments:
        ts = format_timestamp(seg["start"])
        text = seg["text"].strip()
        if text:
            lines.append(f"**[{ts}]** {text}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
