"""
Shared helper functions and constants used across the whole pipeline
(not specific to any single Extract / Transform / Load stage).
"""
import shutil

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


def check_ffmpeg_available():
    """Check that ffmpeg/ffprobe are available on PATH before running the pipeline."""
    for binary in (FFMPEG_BIN, FFPROBE_BIN):
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"Could not find '{binary}' on PATH.\n"
                "Please install FFmpeg (see README.md) and try again."
            )


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
