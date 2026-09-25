"""
Unit tests for process/pipeline/enrich/diarize.py: assign_speakers() (the
pure interval-overlap logic that attaches a speaker label to each Whisper
segment) plus, since a real bug was found here running this project live
(see CODE_REVIEW.md), diarize_audio()'s Pipeline.from_pretrained()
token=/use_auth_token= compatibility fallback - with a fake pyannote.audio
module injected via sys.modules, the same technique
test_correct_transcript.py uses for `anthropic`. The real pyannote.audio
model call itself is still not covered here - see process/tests/README.md
for why.

Run: python -m unittest tests.unit.test_diarize -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.enrich.diarize import assign_speakers, diarize_audio


def _seg(start, end):
    return {"start": start, "end": end, "text": "..."}


def _silent_log(_message):
    pass


def _fake_pipeline_instance(turns, wrap_in_diarize_output=False):
    """
    A stand-in for the object Pipeline.from_pretrained() returns: calling
    it returns a fake diarization result whose .itertracks(yield_label=True)
    yields (turn, _, speaker) the same shape pyannote's real Annotation does.

    `wrap_in_diarize_output=True` simulates pyannote.audio 4.x, which wraps
    that Annotation inside a DiarizeOutput dataclass's `.speaker_diarization`
    attribute instead of returning it directly (a real bug hit running this
    project live - see CODE_REVIEW.md). Default False simulates pre-4.0,
    where the call result IS the Annotation - no `.speaker_diarization` at
    all, which is why it's deleted below rather than just left unset (a
    bare MagicMock auto-creates any attribute you access, so leaving it
    alone would silently defeat diarize_audio()'s getattr(..., default)
    fallback and not actually test the pre-4.0 code path).
    """
    diarization_result = MagicMock()

    def itertracks(yield_label=True):
        for start, end, speaker in turns:
            yield MagicMock(start=start, end=end), None, speaker

    diarization_result.itertracks.side_effect = itertracks

    if wrap_in_diarize_output:
        call_result = MagicMock()
        call_result.speaker_diarization = diarization_result
    else:
        call_result = diarization_result
        del call_result.speaker_diarization

    instance = MagicMock()
    instance.return_value = call_result  # pipeline(...) -> call_result
    # Real pyannote/torch .to(device) mutates in place and returns self, so
    # diarize_audio()'s `pipeline = pipeline.to(device)` must keep pointing
    # at this same configured instance, not an unconfigured child mock.
    instance.to.return_value = instance
    return instance


def _fake_pyannote_audio_module(from_pretrained_mock):
    fake_pyannote = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = MagicMock()
    fake_audio.Pipeline.from_pretrained = from_pretrained_mock
    return fake_pyannote, fake_audio


class _FakeDevice:
    """Stand-in for a real torch.device - just needs a readable .type."""

    def __init__(self, type_str):
        self.type = type_str

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and self.type == other.type

    def __repr__(self):
        return f"_FakeDevice({self.type!r})"


def _fake_torch_module(cuda_available=False):
    """
    Minimal stand-in for `torch` - diarize_audio() only needs
    torch.cuda.is_available() and torch.device(str). Real torch is not
    installed in every environment this suite runs in (see
    process/tests/README.md), so this avoids requiring it just to exercise
    diarize_audio()'s GPU-selection branch.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.device = _FakeDevice
    fake_torch.from_numpy = lambda arr: arr  # identity is enough - tests only inspect what's passed onward
    return fake_torch


def _fake_soundfile_module(data=None, sample_rate=16000):
    """
    Minimal stand-in for `soundfile` - diarize_audio() only needs
    sf.read(path, dtype="float32", always_2d=True) -> (numpy array,
    sample_rate). Real soundfile is not installed in every environment
    this suite runs in (see process/tests/README.md). A real numpy array
    is used (not a MagicMock) because diarize_audio() calls .T on it.
    """
    fake_sf = types.ModuleType("soundfile")
    fake_data = data if data is not None else np.zeros((10, 1), dtype="float32")
    fake_sf.read = MagicMock(return_value=(fake_data, sample_rate))
    return fake_sf


class TestAssignSpeakers(unittest.TestCase):
    def test_segment_fully_inside_one_turn(self):
        segments = [_seg(2.0, 4.0)]
        turns = [(0.0, 10.0, "SPEAKER_00")]
        result = assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")

    def test_segment_spanning_two_turns_picks_larger_overlap(self):
        # Segment [4, 10]: overlaps SPEAKER_00 by 1s (4-5), SPEAKER_01 by 5s (5-10).
        segments = [_seg(4.0, 10.0)]
        turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 12.0, "SPEAKER_01")]
        result = assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_01")

    def test_no_overlapping_turn_leaves_speaker_none(self):
        segments = [_seg(20.0, 22.0)]
        turns = [(0.0, 5.0, "SPEAKER_00")]
        result = assign_speakers(segments, turns)
        self.assertIsNone(result[0]["speaker"])

    def test_touching_boundary_is_not_counted_as_overlap(self):
        # Turn ends exactly where the segment starts - zero real overlap,
        # must not be selected (code only accepts overlap > 0).
        segments = [_seg(5.0, 8.0)]
        turns = [(0.0, 5.0, "SPEAKER_00")]
        result = assign_speakers(segments, turns)
        self.assertIsNone(result[0]["speaker"])

    def test_empty_turns_list_leaves_every_segment_none(self):
        segments = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
        result = assign_speakers(segments, [])
        self.assertIsNone(result[0]["speaker"])
        self.assertIsNone(result[1]["speaker"])

    def test_multiple_segments_are_independent(self):
        segments = [_seg(0.0, 2.0), _seg(2.0, 4.0), _seg(4.0, 6.0)]
        turns = [(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")]
        result = assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_00")  # 2-3 in turn0 (1s) vs 3-4 in turn1 (1s): tie, first wins
        self.assertEqual(result[2]["speaker"], "SPEAKER_01")

    def test_tie_break_keeps_first_seen_turn(self):
        # Segment [2, 4] overlaps turn0 [0,3] by exactly 1s and turn1 [3,6]
        # by exactly 1s too - a true tie. Code uses strict `>`, so the
        # first turn encountered keeps its assignment.
        segments = [_seg(2.0, 4.0)]
        turns = [(0.0, 3.0, "SPEAKER_A"), (3.0, 6.0, "SPEAKER_B")]
        result = assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_A")

    def test_returns_the_same_list_object_mutated_in_place(self):
        segments = [_seg(0.0, 2.0)]
        turns = [(0.0, 2.0, "SPEAKER_00")]
        result = assign_speakers(segments, turns)
        self.assertIs(result, segments)


class TestDiarizeAudioParamCompat(unittest.TestCase):
    """
    Regression tests for the Pipeline.from_pretrained() token=/
    use_auth_token= compatibility fallback in diarize_audio() - a real bug
    found running this project on a real machine with a newer
    pyannote.audio than use_auth_token= supports (see CODE_REVIEW.md).
    """

    def test_uses_token_kwarg_when_supported(self):
        turns_in = [(0.0, 2.0, "SPEAKER_00")]
        from_pretrained = MagicMock(return_value=_fake_pipeline_instance(turns_in))
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)

        fake_torch = _fake_torch_module(cuda_available=False)
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        self.assertEqual(turns, turns_in)
        from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-community-1", token="hf_token"
        )

    def test_falls_back_to_use_auth_token_kwarg_on_typeerror(self):
        turns_in = [(0.0, 2.0, "SPEAKER_00")]
        from_pretrained = MagicMock(
            side_effect=[TypeError("unexpected keyword argument 'token'"), _fake_pipeline_instance(turns_in)]
        )
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)

        fake_torch = _fake_torch_module(cuda_available=False)
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        self.assertEqual(turns, turns_in)
        self.assertEqual(from_pretrained.call_count, 2)
        second_call_kwargs = from_pretrained.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs, {"use_auth_token": "hf_token"})

    def test_a_typeerror_from_the_actual_diarization_call_is_not_mistaken_for_the_fallback(self):
        # The try/except in diarize_audio() only wraps the from_pretrained()
        # call itself - a TypeError raised later, when actually running the
        # pipeline on the audio, must propagate normally, not silently
        # retry from_pretrained() a second time.
        instance = MagicMock(side_effect=TypeError("unrelated failure inside the real pipeline call"))
        instance.to.return_value = instance  # .to(device) keeps pointing at the same instance
        from_pretrained = MagicMock(return_value=instance)
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)

        fake_torch = _fake_torch_module(cuda_available=False)
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            with self.assertRaises(TypeError):
                diarize_audio("audio.wav", "hf_token", _silent_log)

        from_pretrained.assert_called_once()

    def test_multiple_turns_and_speakers_pass_through_correctly(self):
        turns_in = [
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
            (2.0, 3.0, "SPEAKER_00"),
        ]
        from_pretrained = MagicMock(return_value=_fake_pipeline_instance(turns_in))
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)

        fake_torch = _fake_torch_module(cuda_available=False)
        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        self.assertEqual(turns, turns_in)

    def test_moves_pipeline_to_cpu_when_cuda_not_available(self):
        turns_in = [(0.0, 2.0, "SPEAKER_00")]
        from_pretrained = MagicMock(return_value=_fake_pipeline_instance(turns_in))
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)
        fake_torch = _fake_torch_module(cuda_available=False)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            diarize_audio("audio.wav", "hf_token", _silent_log)

        pipeline_instance = from_pretrained.return_value
        pipeline_instance.to.assert_called_once_with(_FakeDevice("cpu"))

    def test_moves_pipeline_to_cuda_when_available(self):
        turns_in = [(0.0, 2.0, "SPEAKER_00")]
        from_pretrained = MagicMock(return_value=_fake_pipeline_instance(turns_in))
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)
        fake_torch = _fake_torch_module(cuda_available=True)

        with patch.dict(sys.modules, {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": _fake_soundfile_module()}):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        pipeline_instance = from_pretrained.return_value
        pipeline_instance.to.assert_called_once_with(_FakeDevice("cuda"))
        # GPU move must not break the normal return path.
        self.assertEqual(turns, turns_in)

    def test_pipeline_is_called_with_a_preloaded_waveform_dict_not_a_path(self):
        # Regression test for a real bug hit running this project on
        # Windows: pipeline(audio_path) makes pyannote.audio decode the
        # path internally via torchcodec, which failed to load its DLLs
        # ("Could not load libtorchcodec...", see CODE_REVIEW.md). A first
        # fix (torchaudio.load()) also failed - modern torchaudio routes
        # through torchcodec internally too. soundfile (libsndfile) has no
        # FFmpeg/torchcodec dependency at all, so it's used instead.
        turns_in = [(0.0, 2.0, "SPEAKER_00")]
        from_pretrained = MagicMock(return_value=_fake_pipeline_instance(turns_in))
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)
        fake_torch = _fake_torch_module(cuda_available=False)
        # 3 samples, 2 channels - shape (samples, channels), as soundfile returns.
        fake_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype="float32")
        fake_soundfile = _fake_soundfile_module(data=fake_data, sample_rate=16000)

        with patch.dict(
            sys.modules,
            {"pyannote": fake_pyannote, "pyannote.audio": fake_audio, "torch": fake_torch, "soundfile": fake_soundfile},
        ):
            diarize_audio("audio.wav", "hf_token", _silent_log)

        fake_soundfile.read.assert_called_once_with("audio.wav", dtype="float32", always_2d=True)
        pipeline_instance = from_pretrained.return_value
        pipeline_instance.assert_called_once()
        call_kwargs = pipeline_instance.call_args.args[0]
        self.assertEqual(call_kwargs["sample_rate"], 16000)
        # diarize_audio() transposes to (channels, samples) before passing it on.
        np.testing.assert_array_equal(call_kwargs["waveform"], fake_data.T)

    def test_unwraps_diarize_output_from_pyannote_4x(self):
        # Regression test for a real bug hit running this project live:
        # pyannote.audio 4.x wraps the result in a DiarizeOutput dataclass
        # ('DiarizeOutput' object has no attribute 'itertracks') instead of
        # returning the Annotation directly - see CODE_REVIEW.md.
        turns_in = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
        from_pretrained = MagicMock(
            return_value=_fake_pipeline_instance(turns_in, wrap_in_diarize_output=True)
        )
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)
        fake_torch = _fake_torch_module(cuda_available=False)

        with patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "torch": fake_torch,
                "soundfile": _fake_soundfile_module(),
            },
        ):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        self.assertEqual(turns, turns_in)

    def test_still_works_when_pipeline_returns_the_annotation_directly(self):
        # Companion to the above - locks in that diarize_audio() still
        # supports pre-4.0 pyannote.audio, which returns the Annotation
        # directly with no .speaker_diarization wrapper at all.
        turns_in = [(0.0, 1.0, "SPEAKER_00")]
        from_pretrained = MagicMock(
            return_value=_fake_pipeline_instance(turns_in, wrap_in_diarize_output=False)
        )
        fake_pyannote, fake_audio = _fake_pyannote_audio_module(from_pretrained)
        fake_torch = _fake_torch_module(cuda_available=False)

        with patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "torch": fake_torch,
                "soundfile": _fake_soundfile_module(),
            },
        ):
            turns = diarize_audio("audio.wav", "hf_token", _silent_log)

        self.assertEqual(turns, turns_in)


if __name__ == "__main__":
    unittest.main()
