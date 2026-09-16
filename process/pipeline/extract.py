"""
EXTRACT stage: read metadata and pull raw data (audio) from the source video
using FFmpeg. This is the first step of the ETL pipeline - it only retrieves
data, it does not transform anything yet.
"""
import json
import subprocess

from .utils import FFMPEG_BIN, FFPROBE_BIN


def probe_media_info(video_path: str) -> dict:
    """Read video metadata via ffprobe: duration, whether it has an audio track."""
    cmd = [
        FFPROBE_BIN, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0.0))
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if not has_audio:
        raise RuntimeError("This video file has no audio track to transcribe.")

    return {"duration": duration}


def extract_audio(video_path: str, audio_path: str):
    """Extract audio as mono 16kHz PCM WAV - Whisper's expected input format."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr[-2000:]}")
