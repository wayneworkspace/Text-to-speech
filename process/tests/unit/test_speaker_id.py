"""
Unit tests for process/pipeline/enrich/speaker_id.py: the pure,
non-model-dependent pieces of speaker recognition (matching math, name
resolution, profile storage) plus, since a real bug was found here running
this project live (see CODE_REVIEW.md), the embedding-computation path -
_embed_turns()/compute_speaker_centroids()/enroll_speaker() - with fake
pyannote.audio/pyannote.core/torch/soundfile modules injected via
sys.modules, the same technique test_diarize.py uses for diarize_audio().
The real pyannote.audio model call itself is still not covered here - see
process/tests/README.md.

Run: python -m unittest process.tests.unit.test_speaker_id -v
"""
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from pipeline.enrich import speaker_id


def _silent_log(_message):
    pass


class _FakeSegment:
    """
    Stand-in for pyannote.core.Segment - _embed_turns() only constructs one
    per turn and passes it straight into inference.crop(); equality lets
    tests assert exactly which (start, end) pairs were cropped.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return isinstance(other, _FakeSegment) and self.start == other.start and self.end == other.end

    def __repr__(self):
        return f"_FakeSegment({self.start}, {self.end})"


def _fake_pyannote_modules(inference_instance, model=None):
    """
    Fake pyannote.audio (Model.from_pretrained/Inference) and pyannote.core
    (Segment) modules. _load_embedding_model() calls
    Model.from_pretrained(...) then Inference(model, window="whole") - the
    fake Inference() constructor returns `inference_instance` regardless of
    its arguments, so tests can assert directly against that same object.
    """
    fake_pyannote = types.ModuleType("pyannote")

    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Model = MagicMock()
    fake_audio.Model.from_pretrained = MagicMock(return_value=model or MagicMock())
    fake_audio.Inference = MagicMock(return_value=inference_instance)

    fake_core = types.ModuleType("pyannote.core")
    fake_core.Segment = _FakeSegment

    return fake_pyannote, fake_audio, fake_core


def _fake_torch_module():
    """Minimal stand-in for `torch` - pipeline.utils.load_waveform() only needs torch.from_numpy()."""
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = lambda arr: arr
    return fake_torch


def _fake_soundfile_module(data=None, sample_rate=16000):
    """Minimal stand-in for `soundfile` - pipeline.utils.load_waveform() only needs sf.read(...)."""
    fake_sf = types.ModuleType("soundfile")
    fake_data = data if data is not None else np.zeros((10, 1), dtype="float32")
    fake_sf.read = MagicMock(return_value=(fake_data, sample_rate))
    return fake_sf


def _sys_modules_patch(inference_instance, model=None, soundfile_data=None, sample_rate=16000):
    """
    patch.dict(sys.modules, ...) context manager providing fake
    pyannote.audio / pyannote.core / torch / soundfile modules - enough
    for _load_embedding_model(), _embed_turns(), and
    pipeline.utils.load_waveform() to all run against `inference_instance`
    without needing the real (heavy) dependencies installed.
    """
    fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference_instance, model)
    return patch.dict(
        sys.modules,
        {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "pyannote.core": fake_core,
            "torch": _fake_torch_module(),
            "soundfile": _fake_soundfile_module(soundfile_data, sample_rate),
        },
    )


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(v, v), 1.0, places=6)

    def test_opposite_vectors_score_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), -1.0, places=6)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 0.0, places=6)

    def test_known_45_degree_angle(self):
        # [1,0] vs [1,1]: cos(45 deg) = 1/sqrt(2) - hand-computable, unlike
        # arbitrary vectors, so this pins down the exact expected value
        # instead of just checking "some number between 0 and 1".
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 1 / math.sqrt(2), places=6)

    def test_not_affected_by_magnitude(self):
        a = np.array([1.0, 0.0])
        b = np.array([50.0, 0.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 1.0, places=6)


class TestMatchSpeakers(unittest.TestCase):
    def setUp(self):
        # 45-degree pair -> similarity ~0.7071, deliberately chosen so a
        # threshold of 0.50 accepts it and a threshold of 0.75 rejects it.
        self.centroids = {"SPEAKER_00": np.array([1.0, 1.0])}
        self.profiles = [{"name": "Wayne", "embedding": [1.0, 0.0]}]

    def test_match_above_threshold(self):
        result = speaker_id.match_speakers(self.centroids, self.profiles, threshold=0.50)
        self.assertEqual(result, {"SPEAKER_00": "Wayne"})

    def test_no_match_below_threshold(self):
        result = speaker_id.match_speakers(self.centroids, self.profiles, threshold=0.75)
        self.assertEqual(result, {})  # ~0.7071 does not clear 0.75

    def test_picks_the_closer_of_two_profiles(self):
        centroids = {"SPEAKER_00": np.array([1.0, 0.0])}
        profiles = [
            {"name": "Far", "embedding": [0.0, 1.0]},    # similarity 0.0
            {"name": "Close", "embedding": [0.9, 0.1]},  # similarity ~0.994
        ]
        result = speaker_id.match_speakers(centroids, profiles, threshold=0.5)
        self.assertEqual(result, {"SPEAKER_00": "Close"})

    def test_empty_profiles_yields_no_matches(self):
        result = speaker_id.match_speakers(self.centroids, [], threshold=0.1)
        self.assertEqual(result, {})


class TestResolveSpeakerNames(unittest.TestCase):
    def test_matched_speaker_gets_enrolled_name(self):
        segments = [{"speaker": "SPEAKER_00", "text": "hi"}]
        result = speaker_id.resolve_speaker_names(segments, {"SPEAKER_00": "Wayne"})
        self.assertEqual(result[0]["speaker"], "Wayne")

    def test_unmatched_speakers_fall_back_in_order_of_first_appearance(self):
        segments = [
            {"speaker": "SPEAKER_01", "text": "a"},
            {"speaker": "SPEAKER_00", "text": "b"},
            {"speaker": "SPEAKER_01", "text": "c"},
        ]
        result = speaker_id.resolve_speaker_names(segments, {})
        self.assertEqual(result[0]["speaker"], "Speaker 1")  # SPEAKER_01 seen first
        self.assertEqual(result[1]["speaker"], "Speaker 2")  # SPEAKER_00 seen second
        self.assertEqual(result[2]["speaker"], "Speaker 1")  # SPEAKER_01 again -> same label

    def test_mixed_matched_and_unmatched(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "known"},
            {"speaker": "SPEAKER_01", "text": "unknown"},
        ]
        result = speaker_id.resolve_speaker_names(segments, {"SPEAKER_00": "Wayne"})
        self.assertEqual(result[0]["speaker"], "Wayne")
        self.assertEqual(result[1]["speaker"], "Speaker 1")

    def test_segments_without_speaker_field_are_untouched(self):
        segments = [{"text": "no diarization ran"}]
        result = speaker_id.resolve_speaker_names(segments, {})
        self.assertNotIn("speaker", result[0])


class TestProfileStorage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="speaker_id_test_")
        self.path = os.path.join(self.tmp_dir, "sub", "speaker_profiles.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_missing_file_returns_empty_list(self):
        self.assertEqual(speaker_id.load_profiles(self.path), [])

    def test_save_then_load_roundtrip(self):
        profiles = [{"name": "Wayne", "embedding": [1.0, 2.0], "sample_count": 1}]
        speaker_id.save_profiles(self.path, profiles)
        self.assertEqual(speaker_id.load_profiles(self.path), profiles)


def _fake_pyannote_audio_module_for_embedding(from_pretrained_mock, inference_instance=None):
    """
    Fake pyannote.audio module exposing only Model.from_pretrained and
    Inference - enough for _load_embedding_model()'s token=/
    use_auth_token= compatibility fallback tests below, without needing
    the fuller pyannote.core.Segment setup _fake_pyannote_modules() adds
    for _embed_turns().
    """
    fake_pyannote = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Model = MagicMock()
    fake_audio.Model.from_pretrained = from_pretrained_mock
    fake_audio.Inference = MagicMock(return_value=inference_instance or MagicMock())
    return fake_pyannote, fake_audio


class TestLoadEmbeddingModelParamCompat(unittest.TestCase):
    """
    Regression tests for _load_embedding_model()'s token=/use_auth_token=
    compatibility fallback - the same huggingface_hub rename already
    handled for Pipeline.from_pretrained() in diarize.py (see
    CODE_REVIEW.md, section 9.7), found here too running this project
    live: Model.from_pretrained() on the installed pyannote.audio (4.0.7)
    silently swallows an unrecognized use_auth_token= into **kwargs
    instead of raising, so the token never reaches the HuggingFace Hub
    download call - see CODE_REVIEW.md, section 9.15.
    """

    def test_uses_token_kwarg_when_supported(self):
        fake_model = MagicMock()
        from_pretrained = MagicMock(return_value=fake_model)
        fake_pyannote, fake_audio = _fake_pyannote_audio_module_for_embedding(from_pretrained)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio}):
            speaker_id._load_embedding_model("hf_token")

        from_pretrained.assert_called_once_with(speaker_id._EMBEDDING_MODEL_NAME, token="hf_token")

    def test_falls_back_to_use_auth_token_kwarg_on_typeerror(self):
        fake_model = MagicMock()
        from_pretrained = MagicMock(
            side_effect=[TypeError("unexpected keyword argument 'token'"), fake_model]
        )
        fake_pyannote, fake_audio = _fake_pyannote_audio_module_for_embedding(from_pretrained)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio}):
            speaker_id._load_embedding_model("hf_token")

        self.assertEqual(from_pretrained.call_count, 2)
        second_call_kwargs = from_pretrained.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs, {"use_auth_token": "hf_token"})

    def test_inference_is_constructed_with_the_loaded_model_and_whole_window(self):
        fake_model = MagicMock()
        from_pretrained = MagicMock(return_value=fake_model)
        inference_instance = MagicMock()
        fake_pyannote, fake_audio = _fake_pyannote_audio_module_for_embedding(from_pretrained, inference_instance)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio}):
            result = speaker_id._load_embedding_model("hf_token")

        fake_audio.Inference.assert_called_once_with(fake_model, window="whole")
        self.assertIs(result, inference_instance)


class TestEmbedTurns(unittest.TestCase):
    """
    Regression tests for _embed_turns() - a real bug found running this
    project live (see CODE_REVIEW.md): it used to pass the raw audio_path
    string to inference.crop() on every turn, which makes pyannote.audio
    decode it internally via torchcodec (the same DLL-loading failure
    diarize_audio() hit on Windows - see CODE_REVIEW.md and
    test_diarize.py). It now takes a pre-loaded
    {"waveform", "sample_rate"} dict instead.
    """

    def test_crop_receives_the_preloaded_audio_object_not_a_path(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        audio = {"waveform": "preloaded", "sample_rate": 16000}
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            speaker_id._embed_turns(inference, audio, [(0.0, 3.0)], _silent_log)

        cropped_audio_arg = inference.crop.call_args.args[0]
        self.assertIs(cropped_audio_arg, audio)  # not "audio_path" or any string

    def test_averages_vectors_across_multiple_turns(self):
        inference = MagicMock()
        inference.crop.side_effect = [np.array([1.0, 1.0]), np.array([3.0, 3.0])]
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            result = speaker_id._embed_turns(inference, {}, [(0.0, 3.0), (5.0, 8.0)], _silent_log)

        np.testing.assert_array_equal(result, np.array([2.0, 2.0]))

    def test_skips_turns_shorter_than_the_minimum_when_a_longer_one_exists(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([9.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        # second turn is 0.5s - below _MIN_TURN_SECONDS_FOR_EMBEDDING (1.5s)
        turns = [(0.0, 3.0), (10.0, 10.5)]
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            speaker_id._embed_turns(inference, {}, turns, _silent_log)

        self.assertEqual(inference.crop.call_count, 1)
        cropped_segment = inference.crop.call_args.args[1]
        self.assertEqual(cropped_segment, _FakeSegment(0.0, 3.0))

    def test_falls_back_to_all_turns_when_every_turn_is_below_the_minimum(self):
        # If every turn is short, use them anyway rather than embedding nothing.
        inference = MagicMock()
        inference.crop.return_value = np.array([9.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        turns = [(0.0, 0.3), (1.0, 1.2)]
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            speaker_id._embed_turns(inference, {}, turns, _silent_log)

        self.assertEqual(inference.crop.call_count, 2)

    def test_caps_turns_at_the_per_speaker_maximum_longest_first(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        # 6 turns (durations 2..7, all long enough) - only the 5 longest
        # (_MAX_TURNS_PER_SPEAKER) should actually be cropped.
        turns = [(0.0, float(n)) for n in range(2, 8)]
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            speaker_id._embed_turns(inference, {}, turns, _silent_log)

        self.assertEqual(inference.crop.call_count, 5)
        cropped_ends = sorted(call.args[1].end for call in inference.crop.call_args_list)
        self.assertEqual(cropped_ends, [3.0, 4.0, 5.0, 6.0, 7.0])  # the shortest turn (duration 2) was dropped

    def test_one_failing_turn_does_not_lose_the_others(self):
        inference = MagicMock()
        inference.crop.side_effect = [RuntimeError("decode failed"), np.array([2.0, 2.0])]
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        messages = []
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            result = speaker_id._embed_turns(inference, {}, [(0.0, 3.0), (5.0, 8.0)], messages.append)

        np.testing.assert_array_equal(result, np.array([2.0, 2.0]))
        self.assertTrue(any("decode failed" in m for m in messages))

    def test_returns_none_when_every_turn_fails(self):
        inference = MagicMock()
        inference.crop.side_effect = RuntimeError("decode failed")
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "pyannote.core": fake_core}):
            result = speaker_id._embed_turns(inference, {}, [(0.0, 3.0)], _silent_log)

        self.assertIsNone(result)


class TestComputeSpeakerCentroids(unittest.TestCase):
    """
    Regression tests for compute_speaker_centroids(): the waveform must be
    loaded from disk exactly once per video (not once per speaker - a real
    bug this fix could easily reintroduce), and the function must degrade
    to {} rather than raise if either the embedding model or the audio
    itself fails to load (same "never break the pipeline" philosophy as
    diarize.py - see module docstring).
    """

    def test_empty_turns_returns_empty_dict_without_loading_anything(self):
        result = speaker_id.compute_speaker_centroids("audio.wav", [], "hf_token", _silent_log)
        self.assertEqual(result, {})

    def test_computes_one_centroid_per_speaker(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])
        with _sys_modules_patch(inference):
            turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
            result = speaker_id.compute_speaker_centroids("audio.wav", turns, "hf_token", _silent_log)

        self.assertEqual(set(result.keys()), {"SPEAKER_00", "SPEAKER_01"})

    def test_loads_the_audio_file_only_once_regardless_of_speaker_count(self):
        # A real bug this project could easily reintroduce: re-reading the
        # same file from disk once per speaker instead of once per video.
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)
        fake_soundfile = _fake_soundfile_module()

        with patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "pyannote.core": fake_core,
                "torch": _fake_torch_module(),
                "soundfile": fake_soundfile,
            },
        ):
            turns = [
                (0.0, 2.0, "SPEAKER_00"),
                (2.0, 4.0, "SPEAKER_01"),
                (4.0, 6.0, "SPEAKER_02"),
            ]
            speaker_id.compute_speaker_centroids("audio.wav", turns, "hf_token", _silent_log)

        fake_soundfile.read.assert_called_once()

    def test_turns_with_the_same_speaker_label_are_grouped_together(self):
        inference = MagicMock()
        inference.crop.side_effect = [np.array([1.0, 1.0]), np.array([3.0, 3.0])]
        with _sys_modules_patch(inference):
            turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_00")]
            result = speaker_id.compute_speaker_centroids("audio.wav", turns, "hf_token", _silent_log)

        self.assertEqual(inference.crop.call_count, 2)  # both turns embedded for the one speaker
        np.testing.assert_array_equal(result["SPEAKER_00"], np.array([2.0, 2.0]))

    def test_degrades_to_empty_dict_when_the_embedding_model_fails_to_load(self):
        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Model = MagicMock()
        fake_audio.Model.from_pretrained = MagicMock(side_effect=RuntimeError("no HF access"))
        fake_audio.Inference = MagicMock()

        messages = []
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio}):
            turns = [(0.0, 2.0, "SPEAKER_00")]
            result = speaker_id.compute_speaker_centroids("audio.wav", turns, "hf_token", messages.append)

        self.assertEqual(result, {})
        self.assertTrue(any("embedding model" in m for m in messages))

    def test_degrades_to_empty_dict_when_the_audio_file_fails_to_load(self):
        # Regression test for the fix itself: compute_speaker_centroids()
        # now pre-loads the audio (pipeline.utils.load_waveform()) before
        # embedding any speaker - if that fails (e.g. a missing/corrupt
        # file), it must degrade gracefully like every other failure mode
        # here, not raise and crash the whole pipeline.
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])
        fake_pyannote, fake_audio, fake_core = _fake_pyannote_modules(inference)
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.read = MagicMock(side_effect=RuntimeError("could not open file"))

        messages = []
        with patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "pyannote.core": fake_core,
                "torch": _fake_torch_module(),
                "soundfile": fake_soundfile,
            },
        ):
            turns = [(0.0, 2.0, "SPEAKER_00")]
            result = speaker_id.compute_speaker_centroids("audio.wav", turns, "hf_token", messages.append)

        self.assertEqual(result, {})
        self.assertTrue(any("audio" in m for m in messages))
        inference.crop.assert_not_called()


class TestEnrollSpeaker(unittest.TestCase):
    """
    Regression tests for enroll_speaker() - a separate call site (used by
    enroll_speaker.py's CLI) that had the exact same raw-path-to-crop()
    bug as _embed_turns()/compute_speaker_centroids(), plus basic coverage
    of the new-profile vs. merge-into-existing-profile behavior, which had
    no test coverage at all before (see process/tests/README.md).
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="speaker_id_test_")
        self.path = os.path.join(self.tmp_dir, "speaker_profiles.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_crop_receives_the_preloaded_audio_not_the_raw_path(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])

        with _sys_modules_patch(inference):
            speaker_id.enroll_speaker(
                "Wayne", "audio.wav", [(0.0, 3.0)], "hf_token", self.path, _silent_log,
            )

        cropped_audio_arg = inference.crop.call_args.args[0]
        self.assertNotEqual(cropped_audio_arg, "audio.wav")
        self.assertIn("waveform", cropped_audio_arg)

    def test_creates_a_new_profile_when_none_exists(self):
        inference = MagicMock()
        inference.crop.return_value = np.array([1.0, 0.0])

        with _sys_modules_patch(inference):
            speaker_id.enroll_speaker(
                "Wayne", "audio.wav", [(0.0, 3.0)], "hf_token", self.path, _silent_log,
            )

        profiles = speaker_id.load_profiles(self.path)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["name"], "Wayne")
        self.assertEqual(profiles[0]["sample_count"], 1)
        np.testing.assert_allclose(profiles[0]["embedding"], [1.0, 0.0])

    def test_merges_into_an_existing_profile_as_a_weighted_average(self):
        speaker_id.save_profiles(self.path, [
            {"name": "Wayne", "embedding": [0.0, 0.0], "sample_count": 1},
        ])
        inference = MagicMock()
        inference.crop.return_value = np.array([2.0, 0.0])

        with _sys_modules_patch(inference):
            speaker_id.enroll_speaker(
                "Wayne", "audio.wav", [(0.0, 3.0)], "hf_token", self.path, _silent_log,
                update_existing=True,
            )

        profiles = speaker_id.load_profiles(self.path)
        self.assertEqual(len(profiles), 1)  # merged, not appended
        self.assertEqual(profiles[0]["sample_count"], 2)
        # (old[0,0]*1 + new[2,0]) / 2 = [1.0, 0.0]
        np.testing.assert_allclose(profiles[0]["embedding"], [1.0, 0.0])

    def test_update_existing_false_appends_a_second_profile_instead_of_merging(self):
        speaker_id.save_profiles(self.path, [
            {"name": "Wayne", "embedding": [0.0, 0.0], "sample_count": 1},
        ])
        inference = MagicMock()
        inference.crop.return_value = np.array([2.0, 0.0])

        with _sys_modules_patch(inference):
            speaker_id.enroll_speaker(
                "Wayne", "audio.wav", [(0.0, 3.0)], "hf_token", self.path, _silent_log,
                update_existing=False,
            )

        profiles = speaker_id.load_profiles(self.path)
        self.assertEqual(len(profiles), 2)

    def test_raises_when_no_vector_could_be_computed(self):
        inference = MagicMock()
        inference.crop.side_effect = RuntimeError("decode failed")

        with _sys_modules_patch(inference):
            with self.assertRaises(RuntimeError):
                speaker_id.enroll_speaker(
                    "Wayne", "audio.wav", [(0.0, 3.0)], "hf_token", self.path, _silent_log,
                )


if __name__ == "__main__":
    unittest.main()
