"""
DIARIZE stage: figure out "who spoke when" with pyannote.audio, then merge
that speaker timeline with Whisper's transcribed segments by time overlap.

This stage is optional. If HUGGINGFACE_TOKEN is not set in .env, main.py
skips calling this module entirely and every segment is left without a
"speaker" field - the rest of the pipeline still works normally.

Speaker labels (e.g. "SPEAKER_00") are anonymous - pyannote has no idea of
real names, it only tells voices apart from each other.
"""


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
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        use_auth_token=hf_token,
    )

    log("Running speaker diarization (this can take a while on CPU)...")
    diarization = pipeline(audio_path)

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
