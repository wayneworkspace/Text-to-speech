#!/usr/bin/env python3
"""
enroll_speaker.py - one-off CLI to enroll a person's voice into the local
speaker-profile database (data/speaker_profiles.json by default), so future
runs of main.py can recognize them by name instead of showing "Speaker N".

Two ways to use it:

1. From a short, clean sample recording of just this one person (recommended,
   most accurate - 20-60s, minimal background noise):

       python process/enroll_speaker.py sample.wav "Nguyen Van A"

   The whole file is treated as one turn belonging to this person.

2. From an already-processed video, picking out one of the diarized
   speakers by its raw label (visible in that run's log, e.g. SPEAKER_00):

       python process/enroll_speaker.py meeting.mp4 "Nguyen Van A" --speaker SPEAKER_00

   This re-runs diarization on the file to recover the turns, then keeps
   only the ones for that label.

Enrolling the same name again (from a new recording) refines the existing
profile instead of replacing it (see pipeline/enrich/speaker_id.py:enroll_speaker).
"""
import argparse
import os
import shutil
import sys
import tempfile

from config import CONFIG
from pipeline.enrich.diarize import diarize_audio
from pipeline.enrich.speaker_id import enroll_speaker
from pipeline.extract.extract import extract_audio, probe_media_info
from pipeline.utils import check_ffmpeg_available


def _log(message: str) -> None:
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a speaker's voice for automatic recognition.")
    parser.add_argument("input_path", help="A video or audio file containing this person's voice.")
    parser.add_argument("name", help="The person's name, exactly as you want it to appear in transcripts.")
    parser.add_argument(
        "--speaker", default=None,
        help="If given, diarize the input first and enroll only this raw label's turns "
             "(e.g. SPEAKER_00, as seen in a previous run's log). If omitted, the whole "
             "file is treated as a clean sample of just this one person's voice.",
    )
    args = parser.parse_args()

    if not CONFIG["huggingface_token"]:
        sys.exit("HUGGINGFACE_TOKEN is not set in .env - required to load the embedding model.")

    check_ffmpeg_available()

    tmp_dir = tempfile.mkdtemp(prefix="enroll_")
    audio_path = os.path.join(tmp_dir, "audio.wav")
    try:
        info = probe_media_info(args.input_path)
        extract_audio(
            args.input_path, audio_path,
            duration=info["duration"],
            chunk_seconds=CONFIG["extract_chunk_seconds"],
            max_workers=CONFIG["extract_max_workers"],
        )

        if args.speaker:
            turns = diarize_audio(audio_path, CONFIG["huggingface_token"], _log)
            person_turns = [(s, e) for s, e, speaker in turns if speaker == args.speaker]
            if not person_turns:
                sys.exit(f"No turns found for speaker label '{args.speaker}' in this file.")
        else:
            person_turns = [(0.0, info["duration"])]

        enroll_speaker(
            args.name, audio_path, person_turns,
            CONFIG["huggingface_token"], CONFIG["speaker_profiles_path"], _log,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Done - '{args.name}' is now enrolled in {CONFIG['speaker_profiles_path']}")


if __name__ == "__main__":
    main()
