#!/usr/bin/env python3
"""
try_pipeline.py - run the real, full run_pipeline() (main.py) against one
video from the command line, without going through the Tkinter GUI.

Same EXTRACT -> TRANSFORM -> ENRICH -> LOAD pipeline main.py's GUI calls -
just with a plain print() as the log callback instead of a GUI progress box.
Needs torch + openai-whisper installed (see README.md "Setup" /
"Installing FFmpeg" - the CPU-only torch install line if you have no
NVIDIA GPU). Uses whatever HUGGINGFACE_TOKEN / ANTHROPIC_API_KEY is set (or
not set) in .env - same as a normal run.

Usage:
    python try_pipeline.py "path/to/video.mp4"
"""
import sys
import time

from main import run_pipeline


def main():
    if len(sys.argv) != 2:
        print('Usage: python try_pipeline.py "path/to/video.mp4"')
        sys.exit(1)

    video_path = sys.argv[1]
    started = time.time()

    def log(msg):
        elapsed = time.time() - started
        print(f"[{elapsed:7.1f}s] {msg}")

    output_path = run_pipeline(video_path, log)
    print(f"\nDone in {time.time() - started:.1f}s. Transcript: {output_path}")


if __name__ == "__main__":
    main()
