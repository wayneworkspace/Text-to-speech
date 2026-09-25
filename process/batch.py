#!/usr/bin/env python3
"""
batch.py - CLI entry point for processing many videos unattended (e.g. from
a Windows Task Scheduler job running every 30-60 minutes), instead of
picking one file at a time through main.py's GUI.

What this adds on top of main.py's run_pipeline(), all backed by
pipeline/batch_state.py:

  - Scans INPUT_DIR recursively for video files and processes any that are
    not already "done" yet.
  - A lock file so two batch.py runs never overlap and fight over the GPU -
    and recovers on its own if a previous run crashed and left it behind.
  - A per-video state table so an interrupted run resumes correctly next
    time instead of silently skipping (or endlessly retrying) unfinished
    videos.
  - Skips a file that is still being copied into INPUT_DIR (checks its size
    is stable across a short delay before touching it).
  - Cleans up orphaned temp folders a crashed run may have left behind.
  - A per-run log file plus one log file per video, so a single failure
    among many videos is easy to find without reading everything.

See CODE_REVIEW.md, sections 7-9, for the full design.

Usage:
    python process/batch.py
"""
import os
import shutil
import tempfile
import time
from datetime import datetime

from config import CONFIG
from main import run_pipeline
from pipeline.batch_state import (
    acquire_lock,
    load_state,
    mark_done,
    mark_failed,
    mark_running,
    recover_interrupted,
    release_lock,
    save_state,
    should_process,
)
from pipeline.utils import check_ffmpeg_available

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

# How long a file's size must stay unchanged before we consider it fully
# copied into INPUT_DIR and safe to start processing.
_STABILITY_CHECK_SECONDS = 3.0


def _make_file_logger(*log_paths: str):
    """Returns a log(message) function that timestamps and appends to every given file, and also prints to stdout."""
    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        for path in log_paths:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    return log


def _is_file_stable(path: str, wait_seconds: float = _STABILITY_CHECK_SECONDS) -> bool:
    """True if the file's size does not change across a short delay - a cheap
    guard against picking up a video that is still being copied in."""
    try:
        size_before = os.path.getsize(path)
        time.sleep(wait_seconds)
        size_after = os.path.getsize(path)
        return size_before == size_after and size_after > 0
    except OSError:
        return False


def _clean_orphaned_temp_dirs(log) -> None:
    """
    A hard crash (killed process, power loss) can leave tempfile.mkdtemp()
    directories from an interrupted run behind, since the `finally:
    shutil.rmtree(...)` in main.py's run_pipeline never got a chance to run.
    Each one can hold a full extracted audio.wav, so over many interrupted
    runs these add up to real disk space. Sweep them at the start of every
    batch run.
    """
    base = tempfile.gettempdir()
    removed = 0
    for name in os.listdir(base):
        if name.startswith("video2text_") or name.startswith("enroll_"):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    if removed:
        log(f"Cleaned up {removed} orphaned temp folder(s) from a previous interrupted run.")


def find_video_files(input_dir: str) -> list:
    """Recursively find video files under input_dir, sorted for stable, predictable ordering."""
    found = []
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def run_batch() -> None:
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    batch_log_path = os.path.join(
        CONFIG["log_dir"], f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log = _make_file_logger(batch_log_path)

    if not CONFIG["input_dir"]:
        log("INPUT_DIR is not set in .env - nothing to do.")
        return

    if not acquire_lock(CONFIG["lock_path"], log):
        log("Another batch.py run is already in progress - exiting.")
        return

    try:
        check_ffmpeg_available()
        _clean_orphaned_temp_dirs(log)

        state = load_state(CONFIG["state_path"])
        state = recover_interrupted(state, log)
        save_state(CONFIG["state_path"], state)

        videos = find_video_files(CONFIG["input_dir"])
        log(f"Found {len(videos)} video file(s) under {CONFIG['input_dir']}.")

        for video_path in videos:
            video_name = os.path.relpath(video_path, CONFIG["input_dir"])

            if not should_process(state, video_name, log):
                continue

            if not _is_file_stable(video_path):
                log(f"Skipping '{video_name}' this run - looks like it is still being copied in.")
                continue

            safe_name = video_name.replace(os.sep, "__")
            per_file_log_path = os.path.join(CONFIG["log_dir"], "per_file", f"{safe_name}.log")
            file_log = _make_file_logger(batch_log_path, per_file_log_path)

            mark_running(state, video_name)
            save_state(CONFIG["state_path"], state)

            try:
                file_log(f"Starting: {video_name}")
                output_path = run_pipeline(video_path, file_log)
                mark_done(state, video_name, output_path)
                file_log(f"Done: {video_name} -> {output_path}")
            except Exception as exc:  # noqa: BLE001 - one bad video must never stop the whole batch
                mark_failed(state, video_name, str(exc))
                file_log(f"FAILED: {video_name} - {exc}")

            save_state(CONFIG["state_path"], state)

        log("Batch run finished.")
    finally:
        release_lock(CONFIG["lock_path"])


if __name__ == "__main__":
    run_batch()
