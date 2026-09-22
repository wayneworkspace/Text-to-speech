"""
TRANSFORM stage: turn raw audio into timestamped text segments.
Two transformations happen here: (1) split the audio into chunks at silence
boundaries, and (2) transcribe each chunk with Whisper, then re-offset the
timestamps so they line up with the original video.
"""
import os
import re
import subprocess

from ..utils import FFMPEG_BIN, format_timestamp

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")

# Same reasoning as extract.py - generous timeouts, just to fail loudly
# instead of hanging forever on a corrupt/unusual input.
_SILENCE_DETECT_TIMEOUT_SECONDS = 3600
_CUT_CHUNK_TIMEOUT_SECONDS = 120

# Whisper models already loaded this session, keyed by model_size, so
# switching between videos in the same GUI run does not reload (and
# re-download-check) a multi-GB model every single time. See
# CODE_REVIEW.md, section 3, for why this mattered.
_LOADED_MODELS = {}


def detect_silences(audio_path: str, noise_db: float, min_dur: float) -> list:
    """Use ffmpeg's silencedetect filter to find silent intervals in the audio."""
    cmd = [
        FFMPEG_BIN, "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SILENCE_DETECT_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Silence detection did not finish within {_SILENCE_DETECT_TIMEOUT_SECONDS}s.")
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
    """Cut one [start, end) audio segment into its own file (stream copy - fast & exact since it is PCM)."""
    duration = max(0.05, end - start)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", audio_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_CUT_CHUNK_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Cutting audio chunk did not finish within {_CUT_CHUNK_TIMEOUT_SECONDS}s.")
    if result.returncode != 0:
        raise RuntimeError(f"Cutting audio chunk failed:\n{result.stderr[-2000:]}")


def load_whisper_model(model_size: str, log):
    """
    Load the Whisper model (runs locally/offline), reusing an already-loaded
    model of the same size instead of reloading it from disk every time this
    is called - matters when processing several videos in one GUI session.
    Imported lazily so the GUI appears instantly.
    """
    if model_size in _LOADED_MODELS:
        log(f"Reusing already-loaded Whisper model '{model_size}'.")
        return _LOADED_MODELS[model_size]

    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    log(f"Loading Whisper model '{model_size}' (first run may take a few minutes to download)...")
    model = whisper.load_model(model_size, device=device)
    log("Model loaded.")

    _LOADED_MODELS[model_size] = model
    return model


def transcribe_chunk(model, chunk_path: str, language, initial_prompt) -> list:
    """
    Transcribe one chunk, returning Whisper's raw segments (start/end/text/
    avg_logprob/... relative to 0). `initial_prompt` is an optional domain
    vocabulary hint (see config.py's DOMAIN_VOCABULARY) that nudges Whisper
    toward correctly spelling technical terms, names, or jargon.
    """
    # fp16 only makes sense (and is only supported) on a CUDA GPU; on CPU,
    # asking for it just makes Whisper silently fall back to fp32 with a
    # warning printed on every single chunk. Decide it explicitly instead.
    use_fp16 = next(model.parameters()).device.type == "cuda"
    result = model.transcribe(
        chunk_path,
        language=language,
        initial_prompt=initial_prompt,
        fp16=use_fp16,
        verbose=False,
    )
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

        segments = transcribe_chunk(
            model, chunk_path, config["language"], config["domain_vocabulary"]
        )
        for seg in segments:
            all_segments.append({
                "start": seg["start"] + start,
                "end": seg["end"] + start,
                "text": seg["text"],
                # Whisper's own confidence score for this segment; flag it if
                # it falls below the configured threshold so the user knows
                # which lines are worth double-checking.
                "low_confidence": seg.get("avg_logprob", 0.0) < config["confidence_threshold"],
            })
        log(f"[{idx}/{len(chunks)}] Done ({len(segments)} sentence(s)).")

    return all_segments
