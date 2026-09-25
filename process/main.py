#!/usr/bin/env python3
"""
main.py - Orchestrator for the ETL pipeline: Extract -> Transform -> Load.

This module does not do any heavy lifting itself - it just calls the
functions in pipeline/extract/, pipeline/transform/, pipeline/load/
in the right order, and starts the GUI (pipeline/gui.py), passing it the
run_pipeline function as a callback.

Usage:
    python main.py
    python main.py "path/to/video.mp4"
"""
import os
import shutil
import sys
import tempfile

from config import CONFIG, PROCESSED_DIR, RAW_AUDIO_DIR
from pipeline.enrich.correct import correct_transcript_errors
from pipeline.enrich.diarize import assign_speakers, diarize_audio
from pipeline.enrich.enrich import segment_topics
from pipeline.enrich.speaker_id import (
    compute_speaker_centroids,
    load_profiles,
    match_speakers,
    resolve_speaker_names,
)
from pipeline.extract.extract import extract_audio, probe_media_info
from pipeline.load.load import write_markdown
from pipeline.transform.transform import load_whisper_model, transform_to_segments
from pipeline.utils import check_ffmpeg_available


def run_pipeline(video_path: str, log) -> str:
    """Orchestrate the full ETL pipeline for one video, returning the markdown file path."""
    check_ffmpeg_available()

    video_path = os.path.abspath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # The raw extracted audio is a real, reusable output now, not a temp
    # file - saved under output/raw/, keyed by video name only (not date):
    # re-running the same video overwrites it, since it's the exact same
    # audio either way - unlike the Markdown transcript below, nothing
    # about a later run changes what a video sounds like, so dating this
    # copy too would just pile up duplicate multi-hundred-MB files.
    audio_path = os.path.join(RAW_AUDIO_DIR, f"{video_name}.wav")

    # Whisper's own chunk_*.wav files (created inside TRANSFORM) are the
    # only thing that still lives here - a true, short-lived temp dir.
    tmp_dir = tempfile.mkdtemp(prefix="video2text_")

    try:
        # ------------------------------------------------------------------
        # EXTRACT: read metadata + pull raw audio from the source video (pipeline/extract/)
        # ------------------------------------------------------------------
        log(f"Reading video info: {video_path}")
        info = probe_media_info(video_path)
        log(f"Video duration: {info['duration']:.0f}s")

        log("Extracting audio (FFmpeg)...")
        extract_audio(
            video_path, audio_path,
            duration=info["duration"],
            chunk_seconds=CONFIG["extract_chunk_seconds"],
            max_workers=CONFIG["extract_max_workers"],
        )

        # ------------------------------------------------------------------
        # TRANSFORM: split into chunks at silence + transcribe speech (pipeline/transform/)
        # ------------------------------------------------------------------
        model = load_whisper_model(CONFIG["model_size"], log)
        segments = transform_to_segments(
            audio_path, info["duration"], CONFIG, model, log, tmp_dir
        )

        # Transcript correction (pipeline/enrich/correct.py) - optional, needs
        # ANTHROPIC_API_KEY. Fixes likely mis-transcribed proper nouns/technical
        # terms before anything downstream (diarization, topics) reads the text.
        if CONFIG["anthropic_api_key"]:
            segments = correct_transcript_errors(
                segments, video_path, CONFIG["domain_vocabulary"],
                CONFIG["anthropic_api_key"], CONFIG["anthropic_model"], log,
                batch_size=CONFIG["correction_batch_size"],
            )
        else:
            log("No ANTHROPIC_API_KEY set - skipping transcript correction.")

        # Speaker diarization + recognition - optional, needs HUGGINGFACE_TOKEN.
        # Wrapped in try/except so it follows the same never-crash-the-pipeline
        # rule as transcript correction/topic segmentation above/below: a bad
        # token, no network, or a pyannote/huggingface_hub version mismatch
        # (this has happened - see CODE_REVIEW.md) must not throw away the
        # transcription Whisper already did. A failure here just means the
        # transcript comes out without speaker labels, same as if
        # HUGGINGFACE_TOKEN had never been set.
        if CONFIG["huggingface_token"]:
            try:
                turns = diarize_audio(audio_path, CONFIG["huggingface_token"], log)
                segments = assign_speakers(segments, turns)

                # Try to recognize enrolled speakers (pipeline/enrich/speaker_id.py). This
                # only maps raw SPEAKER_XX labels to real names or "Speaker N" -
                # it never fails the pipeline; a lookup problem just means every
                # voice falls back to "Speaker 1", "Speaker 2"... (see that
                # module's docstring).
                profiles = load_profiles(CONFIG["speaker_profiles_path"])
                centroids = compute_speaker_centroids(audio_path, turns, CONFIG["huggingface_token"], log)
                name_map = match_speakers(centroids, profiles, CONFIG["speaker_match_threshold"])
                segments = resolve_speaker_names(segments, name_map)
                if name_map:
                    recognized = ", ".join(sorted(set(name_map.values())))
                    log(f"Recognized {len(name_map)} known speaker(s): {recognized}")
            except Exception as exc:
                log(f"Speaker diarization skipped (failed: {exc})")
        else:
            log("No HUGGINGFACE_TOKEN set - skipping speaker diarization.")

        # Topic segmentation (pipeline/enrich/enrich.py) - optional, needs ANTHROPIC_API_KEY.
        if CONFIG["anthropic_api_key"]:
            topics = segment_topics(segments, CONFIG["anthropic_api_key"], CONFIG["anthropic_model"], log)
        else:
            log("No ANTHROPIC_API_KEY set - skipping topic segmentation.")
            topics = []

        # ------------------------------------------------------------------
        # LOAD: write the result to a Markdown file in output/processed/ (pipeline/load/)
        # ------------------------------------------------------------------
        log("Writing Markdown file...")
        output_path = write_markdown(PROCESSED_DIR, video_path, segments, topics)
        log(f"Done! Result: {output_path}")
        return output_path

    finally:
        if not CONFIG["keep_temp_files"]:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            log(f"(Temp files kept at: {tmp_dir})")


def main():
    # Imported here, not at module top-level, so this module (and
    # run_pipeline() specifically) can be imported and unit-tested on a
    # machine/CI without tkinter installed - only actually running the GUI
    # needs it (pipeline/gui.py itself also imports tkinter). No behavior
    # change for normal `python main.py` usage.
    import tkinter as tk
    from tkinter import messagebox

    from pipeline.gui import App

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
