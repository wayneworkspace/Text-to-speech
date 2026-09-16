"""
Global configuration for the video-to-text pipeline.
Tunable settings are read from a .env file at the project root (see README.md
for the full list of keys and what they do).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# process/ is the folder containing this file -> one level up is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Directory where markdown transcripts are written (<project root>/output/)
OUTPUT_DIR = str(PROJECT_ROOT / "output")


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
    # diarization model (see README.md - "Getting a HuggingFace token").
    # Leave empty to skip speaker diarization entirely.
    "huggingface_token": _get_optional_str("HUGGINGFACE_TOKEN", None),

    # Anthropic API key + model, used for topic/semantic segmentation
    # (see README.md - "Getting an Anthropic API key"). Leave the key empty
    # to skip topic segmentation entirely.
    "anthropic_api_key": _get_optional_str("ANTHROPIC_API_KEY", None),
    "anthropic_model": _get_str("ANTHROPIC_MODEL", "claude-sonnet-5"),
}
