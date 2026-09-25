"""
SPEAKER_ID stage: recognize previously-enrolled speakers across different
videos, and let the user enroll new ones.

Diarization (pipeline/diarize.py) only tells voices apart *within one run* -
its "SPEAKER_00" / "SPEAKER_01" labels are just cluster IDs, reset on every
video. This module adds a persistent layer on top of that: for each
diarized voice, compute a "voiceprint" (a fixed-length embedding vector)
and compare it against a small local database of previously-enrolled
people (speaker_profiles.json). A close match gets the enrolled name;
anything that does not match gets a stable "Speaker 1", "Speaker 2"...
label instead of the meaningless raw SPEAKER_00.

This stage is optional, same rule as diarization: it needs HUGGINGFACE_TOKEN
and only runs when diarization already ran. Failures here should degrade to
"no name resolved" (segments keep their raw SPEAKER_XX label), never break
the rest of the pipeline - matches the philosophy of diarize.py/enrich.py.

Turns are embedded from a pre-loaded {"waveform", "sample_rate"} dict
(pipeline.utils.load_waveform()), not a bare file path - the exact same
torchcodec DLL-loading failure diarize.py's diarize_audio() hit on Windows
(see CODE_REVIEW.md) shows up here too, since pyannote.audio's
Inference.crop() decodes a path-typed "file" internally via torchcodec
just like Pipeline.__call__() does. The waveform is loaded once per audio
file (in compute_speaker_centroids()/enroll_speaker(), not inside
_embed_turns() itself) and reused for every turn across every speaker,
which also avoids re-decoding the same file from disk on every single
crop() call.

See CODE_REVIEW.md, section 4, for the design this implements.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

from ..utils import load_waveform

# pyannote/wespeaker-voxceleb-resnet34-LM is the embedding model pyannote's
# own diarization pipelines use internally - same ecosystem, same
# HUGGINGFACE_TOKEN already required for diarization, no extra account setup.
_EMBEDDING_MODEL_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Turns shorter than this are unreliable for a voiceprint (too little audio
# to characterize a voice) - skip them when there is a longer turn to use instead.
_MIN_TURN_SECONDS_FOR_EMBEDDING = 1.5

# Cap how many turns we embed per speaker - a handful of the longest turns
# already gives a stable average; embedding every single turn just costs
# more time for no real accuracy gain.
_MAX_TURNS_PER_SPEAKER = 5


def _load_embedding_model(hf_token: str):
    """
    Load the speaker-embedding model (lazy import, same pattern as
    diarize.py). Tries the current `token=` kwarg first, falling back to
    the older `use_auth_token=` name - same huggingface_hub rename and
    same compatibility fallback as Pipeline.from_pretrained() in
    diarize.py (see CODE_REVIEW.md, section 9.7). Model.from_pretrained()
    behaves differently from Pipeline.from_pretrained() on the version
    found running this project live, though: it does NOT raise TypeError
    for an unrecognized use_auth_token= kwarg, it silently swallows it
    into **kwargs instead - so the token never actually reaches the
    HuggingFace Hub download call. That showed up live as a harmless-
    looking "sending unauthenticated requests to the HF Hub" warning
    (still worked, since this particular model is not gated) rather than
    a crash - see CODE_REVIEW.md, section 9.15.
    """
    from pyannote.audio import Inference, Model

    try:
        model = Model.from_pretrained(_EMBEDDING_MODEL_NAME, token=hf_token)
    except TypeError:
        model = Model.from_pretrained(_EMBEDDING_MODEL_NAME, use_auth_token=hf_token)
    return Inference(model, window="whole")


def _embed_turns(inference, audio, turns: list, log):
    """
    Compute one averaged embedding vector from a list of (start, end) turns
    that all belong to the same voice. `audio` is a pre-loaded
    {"waveform", "sample_rate"} dict (pipeline.utils.load_waveform()), not
    a file path - see module docstring. Returns None if nothing usable was
    found.
    """
    from pyannote.core import Segment

    candidates = [(s, e) for s, e in turns if e - s >= _MIN_TURN_SECONDS_FOR_EMBEDDING] or list(turns)
    candidates.sort(key=lambda t: t[1] - t[0], reverse=True)

    vectors = []
    for start, end in candidates[:_MAX_TURNS_PER_SPEAKER]:
        try:
            vector = inference.crop(audio, Segment(start, end))
            vectors.append(np.asarray(vector).reshape(-1))
        except Exception as exc:
            log(f"  (skipped one turn while computing voice embedding: {exc})")

    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def compute_speaker_centroids(audio_path: str, turns: list, hf_token: str, log) -> dict:
    """
    For each speaker label found by diarization, compute one representative
    embedding vector from that speaker's longest turns.

    Returns {speaker_label: np.ndarray} - empty dict if there are no turns,
    the embedding model cannot be loaded, or the audio file cannot be
    pre-loaded (see pipeline.utils.load_waveform()). Never raises.
    """
    if not turns:
        return {}

    try:
        log("Loading speaker embedding model (pyannote.audio, first run downloads it)...")
        inference = _load_embedding_model(hf_token)
    except Exception as exc:
        log(f"Speaker recognition skipped (could not load embedding model: {exc})")
        return {}

    try:
        audio = load_waveform(audio_path)
    except Exception as exc:
        log(f"Speaker recognition skipped (could not load audio for embedding: {exc})")
        return {}

    turns_by_speaker: dict = {}
    for start, end, speaker in turns:
        turns_by_speaker.setdefault(speaker, []).append((start, end))

    centroids = {}
    for speaker, speaker_turns in turns_by_speaker.items():
        vector = _embed_turns(inference, audio, speaker_turns, log)
        if vector is not None:
            centroids[speaker] = vector

    log(f"Computed voice embeddings for {len(centroids)}/{len(turns_by_speaker)} speaker(s).")
    return centroids


def load_profiles(path: str) -> list:
    """Load the enrolled speaker-profile database. Returns [] if it does not exist yet."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("speakers", [])


def save_profiles(path: str, profiles: list) -> None:
    """Write the speaker-profile database back to disk (creates the folder if needed)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"speakers": profiles}, f, ensure_ascii=False, indent=2)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = same direction (same voice), 0.0 = unrelated. Not affected by volume/length."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def match_speakers(centroids: dict, profiles: list, threshold: float) -> dict:
    """
    Compare each diarized speaker's centroid embedding against every
    enrolled profile. Returns {speaker_label: matched_name} - only for
    speakers whose best match clears `threshold`. A speaker with no match
    above threshold is simply absent from the result; the caller (see
    resolve_speaker_names) decides the fallback label for those.
    """
    matches = {}
    for speaker, vector in centroids.items():
        best_name = None
        best_score = threshold
        for profile in profiles:
            score = _cosine_similarity(vector, np.array(profile["embedding"]))
            if score > best_score:
                best_score = score
                best_name = profile["name"]
        if best_name:
            matches[speaker] = best_name
    return matches


def resolve_speaker_names(segments: list, name_map: dict) -> list:
    """
    Replace each segment's raw diarization label ("SPEAKER_00", ...) with
    either its matched enrolled name, or a stable "Speaker 1", "Speaker 2"...
    fallback assigned in order of first appearance in this video. Segments
    without a "speaker" field (diarization skipped, or no overlapping turn)
    are left untouched.
    """
    fallback_labels: dict = {}
    next_number = 1

    for seg in segments:
        raw_label = seg.get("speaker")
        if raw_label is None:
            continue

        if raw_label in name_map:
            seg["speaker"] = name_map[raw_label]
            continue

        if raw_label not in fallback_labels:
            fallback_labels[raw_label] = f"Speaker {next_number}"
            next_number += 1
        seg["speaker"] = fallback_labels[raw_label]

    return segments


def enroll_speaker(
    name: str,
    audio_path: str,
    turns: list,
    hf_token: str,
    profiles_path: str,
    log,
    update_existing: bool = True,
) -> None:
    """
    Add or update a speaker profile: compute one embedding from the given
    (start, end) turns - which must all belong to this one person, e.g. one
    speaker's turns filtered out of an already-diarized video, or a single
    turn spanning a whole clean sample recording - and save it under `name`.

    If a profile for `name` already exists and update_existing is True, the
    new embedding is merged into the existing one as a running average
    weighted by sample_count, so the profile gets steadily more representative
    the more times someone is enrolled from a new recording.
    """
    inference = _load_embedding_model(hf_token)
    audio = load_waveform(audio_path)
    new_vector = _embed_turns(inference, audio, turns, log)
    if new_vector is None:
        raise RuntimeError(
            "Could not compute a voice embedding from the given audio - "
            "are the turns long enough (need at least a couple of seconds)?"
        )

    profiles = load_profiles(profiles_path)
    existing = next((p for p in profiles if p["name"] == name), None)
    now = datetime.now(timezone.utc).isoformat()

    if existing and update_existing:
        old_count = existing.get("sample_count", 1)
        old_vector = np.array(existing["embedding"])
        merged = (old_vector * old_count + new_vector) / (old_count + 1)
        existing["embedding"] = merged.tolist()
        existing["sample_count"] = old_count + 1
        existing["updated_at"] = now
    else:
        profiles.append({
            "name": name,
            "embedding": new_vector.tolist(),
            "sample_count": 1,
            "updated_at": now,
        })

    save_profiles(profiles_path, profiles)
    log(f"Enrolled/updated speaker profile: {name}")
