#!/usr/bin/env python3
"""
try_extract.py - manual smoke test for the EXTRACT stage
(pipeline/extract/extract.py) against a real video file. Only needs
FFmpeg on PATH - no Whisper/torch/pyannote required, so this is safe to
run even before installing the heavier ML dependencies from
requirements.txt, just to confirm FFmpeg + this pipeline's own code can
read your video and pull its audio out correctly.

Usage:
    python try_extract.py "path/to/video.mp4"
"""
import os
import sys

from config import CONFIG, RAW_AUDIO_DIR
from pipeline.extract.extract import extract_audio, probe_media_info
from pipeline.utils import check_ffmpeg_available


def main():
    if len(sys.argv) != 2:
        print('Usage: python try_extract.py "path/to/video.mp4"')
        sys.exit(1)

    video_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    check_ffmpeg_available()

    print(f"Video: {video_path}")
    info = probe_media_info(video_path)
    print(f"Duration: {info['duration']:.1f}s (has an audio track - probe_media_info would have raised otherwise)")

    # Same output/raw/<video_name>.wav path run_pipeline() itself now uses
    # (see main.py) - this script exercises the real, current behavior,
    # not a throwaway temp file.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(RAW_AUDIO_DIR, f"{video_name}.wav")

    print(f"Extracting audio to: {audio_path}")
    print("(this calls the real extract_audio() from the pipeline)...")
    extract_audio(
        video_path, audio_path,
        duration=info["duration"],
        chunk_seconds=CONFIG["extract_chunk_seconds"],
        max_workers=CONFIG["extract_max_workers"],
    )

    if not os.path.isfile(audio_path):
        print("FAILED - extract_audio() did not produce an output file.")
        sys.exit(1)

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    reprobe = probe_media_info(audio_path)
    delta = abs(reprobe["duration"] - info["duration"])

    print(f"Output: {audio_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Re-probed duration: {reprobe['duration']:.1f}s (original: {info['duration']:.1f}s, diff: {delta:.1f}s)")
    print("OK - durations match." if delta < 1.0 else "WARNING - duration mismatch, check the file.")
    print("(This file stays at output/raw/ - it is the real, persistent EXTRACT-stage output now, not a temp file.)")


if __name__ == "__main__":
    main()
