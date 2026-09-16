#!/usr/bin/env python3
"""
main.py - Orchestrator for the ETL pipeline: Extract -> Transform -> Load.

This module does not do any heavy lifting itself - it just calls the
functions in pipeline/extract.py, pipeline/transform.py, pipeline/load.py
in the right order, and starts the GUI (pipeline/gui.py), passing it the
run_pipeline function as a callback.

Usage:
    python main.py
    python main.py "path/to/video.mp4"
"""
import os
import sys
import shutil
import tempfile

import tkinter as tk
from tkinter import messagebox

from config import CONFIG, OUTPUT_DIR
from pipeline.utils import check_ffmpeg_available
from pipeline.extract import probe_media_info, extract_audio
from pipeline.transform import load_whisper_model, transform_to_segments
from pipeline.diarize import diarize_audio, assign_speakers
from pipeline.enrich import segment_topics
from pipeline.load import write_markdown
from pipeline.gui import App


def run_pipeline(video_path: str, log) -> str:
    """Orchestrate the full ETL pipeline for one video, returning the markdown file path."""
    check_ffmpeg_available()

    video_path = os.path.abspath(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="video2text_")
    audio_path = os.path.join(tmp_dir, "audio.wav")

    try:
        # ------------------------------------------------------------------
        # EXTRACT: read metadata + pull raw audio from the source video (pipeline/extract.py)
        # ------------------------------------------------------------------
        log(f"Reading video info: {video_path}")
        info = probe_media_info(video_path)
        log(f"Video duration: {info['duration']:.0f}s")

        log("Extracting audio (FFmpeg)...")
        extract_audio(video_path, audio_path)

        # ------------------------------------------------------------------
        # TRANSFORM: split into chunks at silence + transcribe speech (pipeline/transform.py)
        # ------------------------------------------------------------------
        model = load_whisper_model(CONFIG["model_size"], log)
        segments = transform_to_segments(
            audio_path, info["duration"], CONFIG, model, log, tmp_dir
        )

        # Speaker diarization (pipeline/diarize.py) - optional, needs HUGGINGFACE_TOKEN.
        if CONFIG["huggingface_token"]:
            turns = diarize_audio(audio_path, CONFIG["huggingface_token"], log)
            segments = assign_speakers(segments, turns)
        else:
            log("No HUGGINGFACE_TOKEN set - skipping speaker diarization.")

        # Topic segmentation (pipeline/enrich.py) - optional, needs ANTHROPIC_API_KEY.
        if CONFIG["anthropic_api_key"]:
            topics = segment_topics(segments, CONFIG["anthropic_api_key"], CONFIG["anthropic_model"], log)
        else:
            log("No ANTHROPIC_API_KEY set - skipping topic segmentation.")
            topics = []

        # ------------------------------------------------------------------
        # LOAD: write the result to a Markdown file in output/ (pipeline/load.py)
        # ------------------------------------------------------------------
        log("Writing Markdown file...")
        output_path = write_markdown(OUTPUT_DIR, video_path, segments, topics)
        log(f"Done! Result: {output_path}")
        return output_path

    finally:
        if not CONFIG["keep_temp_files"]:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            log(f"(Temp files kept at: {tmp_dir})")


def main():
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    try:
        check_ffmpeg_available()
    except RuntimeError as exc:
        root.withdraw()
        messagebox.showerror("Missing FFmpeg", str(exc))
        return
    App(root, run_pipeline=run_pipeline, initial_path=initial_path)
    root.mainloop()


if __name__ == "__main__":
    main()
