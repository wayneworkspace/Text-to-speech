"""
EXTRACT stage: read metadata and pull raw data (audio) from the source video
using FFmpeg. This is the first step of the ETL pipeline - it only retrieves
data, it does not transform anything yet.
"""
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from ..utils import FFMPEG_BIN, FFPROBE_BIN

# Safety nets against a hung ffmpeg/ffprobe process (corrupt input, unusual
# codec, a stream that never ends). These are generous on purpose - normal
# runs finish in seconds to low minutes - they only exist to fail loudly
# instead of hanging the whole pipeline forever (see CODE_REVIEW.md, section 2).
_PROBE_TIMEOUT_SECONDS = 30
_EXTRACT_TIMEOUT_SECONDS = 3600

# Below this duration, splitting extraction into parallel chunks would cost
# more (starting several ffmpeg processes, then concatenating) than it could
# ever save - just extract the whole thing in one pass.
_MIN_DURATION_FOR_CHUNKED_EXTRACT_SECONDS = 600  # 10 minutes


def probe_media_info(video_path: str) -> dict:
    """Read video metadata via ffprobe: duration, whether it has an audio track."""
    cmd = [
        FFPROBE_BIN, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe did not finish within {_PROBE_TIMEOUT_SECONDS}s - is the file readable?")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0.0))
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if not has_audio:
        raise RuntimeError("This video file has no audio track to transcribe.")
    if duration <= 0:
        # Some containers (streamed/remuxed files, a few corrupted headers) do
        # not report format.duration. Fail loudly here instead of silently
        # planning a near-empty chunk and producing a near-empty transcript
        # with no error at all (see CODE_REVIEW.md, section 2).
        raise RuntimeError(
            "ffprobe could not determine this video's duration (got 0). "
            "The file may be corrupted or use an unusual container - try "
            "re-exporting/re-muxing it and running again."
        )

    return {"duration": duration}


def _run_ffmpeg_extract(cmd: list) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_EXTRACT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Audio extraction did not finish within {_EXTRACT_TIMEOUT_SECONDS}s.")
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr[-2000:]}")


def _extract_audio_single_pass(video_path: str, audio_path: str, start: float = None, end: float = None) -> None:
    """One ffmpeg call: extract mono 16kHz PCM WAV, optionally only a [start, end) time range."""
    cmd = [FFMPEG_BIN, "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video_path]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", audio_path]
    _run_ffmpeg_extract(cmd)


def _concat_wav_parts(part_paths: list, audio_path: str, tmp_dir: str) -> None:
    """Stitch the parallel-extracted WAV parts back into one continuous file (all same PCM format, so a plain stream copy works)."""
    concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in part_paths:
            # ffmpeg's concat demuxer list format. These are our own temp
            # paths (part_NNN.wav under a controlled tempdir) so no special
            # character escaping is needed beyond normalizing separators.
            f.write("file '{}'\n".format(p.replace("\\", "/")))

    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_EXTRACT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Concatenating extracted audio parts did not finish within {_EXTRACT_TIMEOUT_SECONDS}s.")
    if result.returncode != 0:
        raise RuntimeError(f"Concatenating extracted audio parts failed:\n{result.stderr[-2000:]}")


def extract_audio(
    video_path: str,
    audio_path: str,
    duration: float = None,
    chunk_seconds: int = 600,
    max_workers: int = 4,
) -> None:
    """
    Extract audio as mono 16kHz PCM WAV - Whisper's expected input format.

    For a long video, this splits the work into `chunk_seconds`-long time
    ranges and runs ffmpeg on each range in parallel (up to max_workers at
    once), instead of one single ffmpeg process reading the whole file start
    to finish. ffmpeg's audio decode is single-threaded per process, so on a
    multi-core machine this uses otherwise-idle CPU cores - the GPU is busy
    with Whisper on a *different* video at this point in the pipeline, not
    this step - to cut wall-clock extraction time roughly by a factor of
    max_workers for long recordings.

    Falls back to a single ffmpeg pass when `duration` is not supplied, or
    the video is short enough that parallelizing would not be worth the
    fixed cost of starting several ffmpeg processes and then concatenating.

    Trade-off: each parallel chunk is cut with fast ("-ss before -i") input
    seeking, the same technique pipeline/transform.py's cut_chunk() already
    uses. This can lose a few tens of milliseconds at each chunk boundary -
    the same class of edge case already documented in README.md's
    Limitations for the Whisper chunking step. In practice this is far
    shorter than a spoken word, so it does not meaningfully affect
    transcription quality.
    """
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)

    if duration is None or duration < _MIN_DURATION_FOR_CHUNKED_EXTRACT_SECONDS:
        _extract_audio_single_pass(video_path, audio_path)
        return

    tmp_dir = tempfile.mkdtemp(prefix="extract_chunks_")
    try:
        starts = [s for s in range(0, int(duration) + 1, chunk_seconds) if s < duration]
        ranges = [(s, min(s + chunk_seconds, duration)) for s in starts]
        part_paths = [os.path.join(tmp_dir, f"part_{i:03d}.wav") for i in range(len(ranges))]

        def _extract_part(i: int) -> None:
            start, end = ranges[i]
            _extract_audio_single_pass(video_path, part_paths[i], start=start, end=end)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # list(...) forces every result to be consumed, so an exception
            # raised inside any one worker propagates out here instead of
            # being silently swallowed by the executor.
            list(pool.map(_extract_part, range(len(ranges))))

        _concat_wav_parts(part_paths, audio_path, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
