"""
DIARIZE stage: figure out "who spoke when" with pyannote.audio, then merge
that speaker timeline with Whisper's transcribed segments by time overlap.

This stage is optional. If HUGGINGFACE_TOKEN is not set in .env, main.py
skips calling this module entirely and every segment is left without a
"speaker" field - the rest of the pipeline still works normally.

Speaker labels (e.g. "SPEAKER_00") are anonymous - pyannote has no idea of
real names, it only tells voices apart from each other.

Audio is pre-loaded via pipeline.utils.load_waveform() and passed as a
{"waveform", "sample_rate"} dict rather than a bare file path - a real
issue hit running this project on Windows: passing a path makes
pyannote.audio decode it internally via torchcodec, which needs FFmpeg's
*shared* DLL build (different from the ffmpeg.exe already used elsewhere
in this pipeline for EXTRACT) and failed to load at all ("Could not load
libtorchcodec..."). The first fix tried here was `torchaudio.load()`
instead (also failed - matching torchaudio versions load *through*
torchcodec internally now, so it hit the exact same DLL error one level
down). `soundfile` (libsndfile) is a separate, self-contained audio
library with no FFmpeg dependency at all, and does not touch torchcodec -
safe to rely on here specifically because this function only ever
receives a WAV file this project's own EXTRACT stage produced (16kHz mono
16-bit PCM - see pipeline/extract/extract.py), never an arbitrary
video/compressed format that would need a general decoder. speaker_id.py's
embedding step hit this exact same torchcodec error too (same root cause -
pyannote.audio decodes any path-typed "file" this way, regardless of which
part of the library receives it), so the load_waveform() fix now lives in
pipeline/utils.py and both stages share it instead of duplicating the
soundfile-loading logic. See CODE_REVIEW.md for the full history (both
failed attempts) before changing this again.
"""
from ..utils import load_waveform


def diarize_audio(audio_path: str, hf_token: str, log) -> list:
    """
    Run pyannote.audio on the full extracted audio track (not per-chunk - the
    model needs the whole conversation to tell speakers apart consistently).
    Returns a list of (start, end, speaker_label) speaker turns.

    Requires a HuggingFace token that has accepted the terms of
    https://huggingface.co/pyannote/speaker-diarization-community-1
    (see README.md - "Getting a HuggingFace token").
    """
    log("Loading speaker diarization model (pyannote.audio, first run downloads it)...")
    import torch
    from pyannote.audio import Pipeline

    # huggingface_hub renamed use_auth_token= to token= (pyannote.audio
    # picked this up too) - try the current name first, fall back for
    # whichever pyannote.audio version is actually installed.
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=hf_token,
        )

    # Unlike Whisper (pipeline/transform/transform.py), pyannote.audio does
    # NOT move itself to a GPU automatically - Pipeline.from_pretrained()
    # always loads onto CPU. Move it explicitly when CUDA is available, the
    # same way transform.py already does for Whisper, so a GPU-equipped
    # machine does not leave the GPU idle for this stage (which can take
    # several minutes on CPU alone for a long recording).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    if device.type == "cuda":
        log("Running speaker diarization on GPU...")
    else:
        log("Running speaker diarization (this can take a while on CPU)...")

    # Pass a pre-loaded waveform, not the file path - see module docstring.
    result = pipeline(load_waveform(audio_path))

    # pyannote.audio 4.x wraps the result in a DiarizeOutput dataclass
    # (.speaker_diarization is the actual Annotation - a real 403->torchcodec->
    # this: 3rd real version-drift bug found running this project live, see
    # CODE_REVIEW.md); pre-4.0 pipelines return the Annotation directly, with
    # no .speaker_diarization attribute at all. Handle both without needing
    # to know in advance which is installed, same spirit as the token=/
    # use_auth_token= fallback above.
    diarization = getattr(result, "speaker_diarization", result)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    speaker_count = len({speaker for _, _, speaker in turns})
    log(f"Diarization found {speaker_count} speaker(s) across {len(turns)} turn(s).")
    return turns


def assign_speakers(segments: list, turns: list) -> list:
    """
    Attach a "speaker" label to each Whisper segment (in place), by picking
    whichever speaker turn overlaps it the most in time. A segment that has
    no overlapping turn at all keeps speaker=None.
    """
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for t_start, t_end, speaker in turns:
            overlap = min(seg["end"], t_end) - max(seg["start"], t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        seg["speaker"] = best_speaker
    return segments
