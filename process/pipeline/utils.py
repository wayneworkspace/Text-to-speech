"""
Shared helper functions and constants used across the whole pipeline
(not specific to any single Extract / Transform / Load stage).
"""
import shutil

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


def check_ffmpeg_available():
    """Check that ffmpeg/ffprobe are available on PATH before running the pipeline."""
    for binary in (FFMPEG_BIN, FFPROBE_BIN):
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"Could not find '{binary}' on PATH.\n"
                "Please install FFmpeg (see README.md) and try again."
            )


def log_anthropic_usage(response, log) -> None:
    """
    Log the real input/output token count an Anthropic API call actually
    billed, if the response carries one (it always should - this is
    defensive in case a future/mocked SDK response doesn't). Token cost
    scales with input (the whole transcript is sent each call) while
    output stays small regardless of transcript length, by design - see
    correct.py/enrich.py's sparse-response docstrings.
    """
    usage = getattr(response, "usage", None)
    if usage is not None:
        log(f"Claude usage: {usage.input_tokens} input + {usage.output_tokens} output tokens")


def extract_text_from_anthropic_response(response) -> str:
    """
    Concatenate the text of every text block in an Anthropic API response,
    skipping any non-text block (e.g. a ThinkingBlock, when extended
    thinking is enabled - it can legitimately be response.content[0],
    ahead of the actual text block). Assuming content[0] is always text is
    the bug this exists to avoid - see CODE_REVIEW.md.
    """
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def load_waveform(audio_path: str) -> dict:
    """
    Pre-load an audio file with `soundfile` and return it as the
    {"waveform", "sample_rate"} dict pyannote.audio accepts in place of a
    bare file path (its AudioFile type is str | Path | IOBase | Mapping -
    see pyannote.audio's core/io.py). Passing a path instead makes
    pyannote.audio decode it internally via torchcodec, which needs
    FFmpeg's *shared* DLL build (different from the ffmpeg.exe already
    used elsewhere in this pipeline for EXTRACT) and failed to load at all
    on Windows running this project live ("Could not load
    libtorchcodec..."). Both diarize.py's diarize_audio() and
    speaker_id.py's embedding step hit this exact same error - pyannote.audio
    decodes a path-typed "file" the same way no matter which part of the
    library receives it - so the fix lives here once and both stages share
    it. `soundfile` (libsndfile) is a separate, self-contained audio
    library with no FFmpeg/torchcodec dependency at all - safe to rely on
    here specifically because every caller only ever passes a WAV file
    this project's own EXTRACT stage produced (16kHz mono 16-bit PCM - see
    pipeline/extract/extract.py), never an arbitrary video/compressed
    format that would need a general decoder. See CODE_REVIEW.md, sections
    9.11-9.14, for the full history (including a torchaudio.load() fix
    attempt that turned out not to work) before changing this again.
    """
    import torch
    import soundfile as sf

    # always_2d keeps shape (samples, channels) even for mono; .T gives the
    # (channels, samples) layout torch audio tensors use.
    data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    return {"waveform": waveform, "sample_rate": sample_rate}


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
