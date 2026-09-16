"""
TRANSFORM stage: turn raw audio into timestamped text segments.
Two transformations happen here: (1) split the audio into chunks at silence
boundaries, and (2) transcribe each chunk with Whisper, then re-offset the
timestamps so they line up with the original video.
"""
import os
import re
import subprocess

from .utils import FFMPEG_BIN, format_timestamp

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silences(audio_path: str, noise_db: float, min_dur: float) -> list:
    """Use ffmpeg's silencedetect filter to find silent intervals in the audio."""
    cmd = [
        FFMPEG_BIN, "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(result.stderr)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(result.stderr)]
    return list(zip(starts, ends[: len(starts)]))


def plan_chunks(duration: float, silences: list, target: int, min_len: int, max_len: int) -> list:
    """Plan chunk boundaries, preferring to cut at the silence closest to the target length."""
    if duration <= target:
        return [(0.0, duration)]

    chunks = []
    cursor = 0.0
    while cursor < duration:
        remaining = duration - cursor
        if remaining <= max_len:
            chunks.append((cursor, duration))
            break

        ideal_end = cursor + target
        window_start = cursor + min_len
        window_end = min(cursor + max_len, duration)

        best_point = None
        best_distance = None
        for s_start, s_end in silences:
            midpoint = (s_start + s_end) / 2
            if window_start <= midpoint <= window_end:
                distance = abs(midpoint - ideal_end)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_point = midpoint

        split_at = best_point if best_point is not None else ideal_end
        split_at = min(split_at, duration)
        chunks.append((cursor, split_at))
        cursor = split_at

    return chunks


def cut_chunk(audio_path: str, start: float, end: float, out_path: str):
    """Cut one [start, end) audio segment into its own file (stream copy - fast & exact since it's PCM)."""
    duration = max(0.05, end - start)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", audio_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Cutting audio chunk failed:\n{result.stderr[-2000:]}")


def load_whisper_model(model_size: str, log):
    """Load the Whisper model (runs locally/offline). Imported lazily so the GUI appears instantly."""
    log(f"Loading Whisper model '{model_size}' (first run may take a few minutes to download)...")
    import whisper
    model = whisper.load_model(model_size)
    log("Model loaded.")
    return model


def transcribe_chunk(model, chunk_path: str, language) -> list:
    """Transcribe one chunk, returning a list of segments (start/end/text relative to 0)."""
    result = model.transcribe(chunk_path, language=language, verbose=False)
    return result.get("segments", [])


def transform_to_segments(audio_path: str, duration: float, config: dict, model, log, tmp_dir: str) -> list:
    """
    High-level transform: split the audio into chunks, transcribe each one, and
    re-offset timestamps. This is the only function main.py needs to call for
    the TRANSFORM stage.
    """
    log("Detecting silence to plan sensible chunk boundaries (FFmpeg silencedetect)...")
    silences = detect_silences(
        audio_path,
        config["silence_noise_db"],
        config["silence_min_duration"],
    )
    log(f"Found {len(silences)} silent interval(s).")

    chunks = plan_chunks(
        duration, silences,
        config["chunk_target_seconds"],
        config["chunk_min_seconds"],
        config["chunk_max_seconds"],
    )
    log(f"Split the video into {len(chunks)} chunk(s) to process.")

    all_segments = []
    for idx, (start, end) in enumerate(chunks, start=1):
        log(f"[{idx}/{len(chunks)}] Cutting & transcribing segment "
            f"{format_timestamp(start)} - {format_timestamp(end)}...")
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.wav")
        cut_chunk(audio_path, start, end, chunk_path)

        segments = transcribe_chunk(model, chunk_path, config["language"])
        for seg in segments:
            all_segments.append({
                "start": seg["start"] + start,
                "end": seg["end"] + start,
                "text": seg["text"],
            })
        log(f"[{idx}/{len(chunks)}] Done ({len(segments)} sentence(s)).")

    return all_segments
