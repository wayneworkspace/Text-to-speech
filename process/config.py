"""
Global configuration for the video-to-text pipeline.
Tunable settings are read from a .env file at the project root (see README.md
for the full list of keys and what they do, and .env.example for a template
with no secrets in it).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# process/ is the folder containing this file -> one level up is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Directory where markdown transcripts are written (<project root>/output/)
OUTPUT_DIR = str(PROJECT_ROOT / "output")

# Default location for the enrolled speaker-profile database (see
# pipeline/speaker_id.py). Lives outside output/ because it is not a
# per-run result - it is a small database that persists and grows across runs.
DEFAULT_SPEAKER_PROFILES_PATH = str(PROJECT_ROOT / "data" / "speaker_profiles.json")

# Defaults for process/batch.py (see pipeline/batch_state.py) - a separate
# entry point from main.py's single-file GUI flow, so these are only read
# when batch.py runs, but live here alongside the rest of the config.
DEFAULT_STATE_PATH = str(PROJECT_ROOT / "state" / "batch_state.json")
DEFAULT_LOCK_PATH = str(PROJECT_ROOT / "state" / ".batch.lock")
DEFAULT_LOG_DIR = str(PROJECT_ROOT / "logs")


def _get_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _get_optional_str(key: str, default):
    """Like _get_str, but an explicitly empty value in .env means None (e.g. auto-detect)."""
    value = os.getenv(key)
    if value is None:
        return default
    return value if value != "" else None


def _get_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value not in (None, "") else default


def _get_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value not in (None, "") else default


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


CONFIG = {
    # Whisper model size: tiny, base, small, medium, large-v3
    "model_size": _get_str("MODEL_SIZE", "medium"),

    # Audio language. "vi" = Vietnamese. Leave LANGUAGE empty in .env for auto-detect.
    "language": _get_optional_str("LANGUAGE", "vi"),

    # Target / min / max length of each audio chunk, in seconds.
    "chunk_target_seconds": _get_int("CHUNK_TARGET_SECONDS", 300),
    "chunk_min_seconds": _get_int("CHUNK_MIN_SECONDS", 120),
    "chunk_max_seconds": _get_int("CHUNK_MAX_SECONDS", 420),

    # Silence-detection sensitivity, used to choose chunk boundaries.
    "silence_noise_db": _get_float("SILENCE_NOISE_DB", -30),
    "silence_min_duration": _get_float("SILENCE_MIN_DURATION", 0.5),

    # Keep temporary audio files (audio.wav, chunk_*.wav) around for debugging.
    "keep_temp_files": _get_bool("KEEP_TEMP_FILES", False),

    # Optional hint fed to Whisper (as `initial_prompt`) so it recognizes
    # domain-specific terms, names, or jargon correctly. Leave empty to skip.
    "domain_vocabulary": _get_optional_str("DOMAIN_VOCABULARY", None),

    # Whisper's own confidence score (avg_logprob) for each segment. Segments
    # scoring below this threshold are flagged in the output as low-confidence,
    # so the user knows which lines are worth double-checking.
    "confidence_threshold": _get_float("CONFIDENCE_THRESHOLD", -1.0),

    # HuggingFace access token, required to download the pyannote speaker
    # diarization + embedding models (see README.md - "Getting a HuggingFace
    # token"). Leave empty to skip speaker diarization and recognition entirely.
    "huggingface_token": _get_optional_str("HUGGINGFACE_TOKEN", None),

    # Anthropic API key + model, used for topic/semantic segmentation
    # (see README.md - "Getting an Anthropic API key"). Leave the key empty
    # to skip topic segmentation entirely.
    "anthropic_api_key": _get_optional_str("ANTHROPIC_API_KEY", None),
    "anthropic_model": _get_str("ANTHROPIC_MODEL", "claude-sonnet-5"),

    # Where the enrolled speaker-voice database lives (see pipeline/speaker_id.py).
    # Not tracked by git (see .gitignore) - it is local biometric-adjacent data.
    "speaker_profiles_path": _get_str("SPEAKER_PROFILES_PATH", DEFAULT_SPEAKER_PROFILES_PATH),

    # Cosine-similarity threshold (0-1) above which a diarized voice is
    # considered a match for an enrolled profile. Lower = more speakers get
    # recognized but more false matches; higher = stricter, more speakers
    # fall back to "Speaker 1", "Speaker 2"... Needs tuning on real data.
    "speaker_match_threshold": _get_float("SPEAKER_MATCH_THRESHOLD", 0.75),

    # If true, a confident match also updates that speaker's stored profile
    # with the new sample (running average) - improves accuracy over time.
    # Off by default: a wrong match while this is on would slowly corrupt
    # the profile, so turn it on only once you trust the matches you are getting.
    "auto_update_speaker_profiles": _get_bool("AUTO_UPDATE_SPEAKER_PROFILES", False),

    # How many seconds of video each parallel ffmpeg worker extracts at once
    # for a long video (see pipeline/extract/extract.py:extract_audio). Only
    # kicks in past a length where parallelizing is actually worth it.
    "extract_chunk_seconds": _get_int("EXTRACT_CHUNK_SECONDS", 600),

    # Max number of ffmpeg extraction workers running at once. Keep this at
    # or below your CPU's core count - each worker is single-threaded, more
    # workers than cores just adds contention, not speed.
    "extract_max_workers": _get_int("EXTRACT_MAX_WORKERS", 4),

    # process/batch.py only - where a batch run looks for videos to process.
    # None (the default) means batch.py has nothing to do until this is set.
    "input_dir": _get_optional_str("INPUT_DIR", None),

    # process/batch.py only - per-video progress table (resume support) and
    # the lock file that keeps two batch.py instances from running at once.
    "state_path": _get_str("STATE_PATH", DEFAULT_STATE_PATH),
    "lock_path": _get_str("LOCK_PATH", DEFAULT_LOCK_PATH),

    # process/batch.py only - where per-run and per-video log files are written.
    "log_dir": _get_str("LOG_DIR", DEFAULT_LOG_DIR),
}


def _validate_chunk_settings(cfg: dict) -> None:
    """
    Fail fast and clearly if .env has nonsensical chunk sizes, instead of
    letting pipeline/transform.py's plan_chunks() silently misbehave later
    (see CODE_REVIEW.md, section 2).
    """
    min_s = cfg["chunk_min_seconds"]
    target_s = cfg["chunk_target_seconds"]
    max_s = cfg["chunk_max_seconds"]
    if not (0 < min_s <= target_s <= max_s):
        raise ValueError(
            "Invalid chunk settings in .env: need "
            "0 < CHUNK_MIN_SECONDS <= CHUNK_TARGET_SECONDS <= CHUNK_MAX_SECONDS "
            f"(got min={min_s}, target={target_s}, max={max_s})."
        )


_validate_chunk_settings(CONFIG)
