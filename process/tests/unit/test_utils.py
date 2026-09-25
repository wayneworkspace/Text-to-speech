"""
Unit tests for pipeline/utils.py's extract_text_from_anthropic_response(),
log_anthropic_usage(), and load_waveform() (the soundfile-based audio
pre-loading helper shared by diarize.py and speaker_id.py - see
CODE_REVIEW.md for the real torchcodec DLL-loading bug this exists to
avoid, and test_diarize.py for the regression test covering the caller
side of this). format_timestamp() is already covered in
test_transform_chunking.py, and check_ffmpeg_available() needs a real PATH
check, exercised via the integration suite instead.

Run: python -m unittest tests.unit.test_utils -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.utils import (
    extract_text_from_anthropic_response,
    load_waveform,
    log_anthropic_usage,
)


def _fake_torch_module():
    """Minimal stand-in for `torch` - load_waveform() only needs torch.from_numpy()."""
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = lambda arr: arr  # identity is enough - tests only inspect what's passed onward
    return fake_torch


def _fake_soundfile_module(data, sample_rate=16000):
    """Minimal stand-in for `soundfile` - load_waveform() only needs sf.read(...)."""
    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(return_value=(data, sample_rate))
    return fake_sf


def _response(*blocks):
    """blocks: (type, text) pairs, mimicking Anthropic response.content."""
    resp = MagicMock()
    resp.content = [MagicMock(type=t, text=text) for t, text in blocks]
    return resp


class TestExtractTextFromAnthropicResponse(unittest.TestCase):
    def test_single_text_block(self):
        resp = _response(("text", "hello"))
        self.assertEqual(extract_text_from_anthropic_response(resp), "hello")

    def test_thinking_block_before_text_block_is_skipped(self):
        # The actual bug this function fixes: a ThinkingBlock (extended
        # thinking) can legitimately be response.content[0], ahead of the
        # real text block - assuming content[0] is always text crashed on
        # this in production (see CODE_REVIEW.md).
        resp = _response(("thinking", None), ("text", "hello"))
        self.assertEqual(extract_text_from_anthropic_response(resp), "hello")

    def test_multiple_text_blocks_are_concatenated(self):
        resp = _response(("text", "hello "), ("text", "world"))
        self.assertEqual(extract_text_from_anthropic_response(resp), "hello world")

    def test_no_text_block_returns_empty_string(self):
        resp = _response(("thinking", None))
        self.assertEqual(extract_text_from_anthropic_response(resp), "")

    def test_block_missing_type_attribute_is_treated_as_non_text(self):
        resp = MagicMock()
        block = MagicMock(spec=["text"])  # accessing .type raises AttributeError
        block.text = "should be ignored"
        resp.content = [block]
        self.assertEqual(extract_text_from_anthropic_response(resp), "")


class TestLoadWaveform(unittest.TestCase):
    """
    Regression tests for load_waveform() - extracted from diarize_audio()
    into pipeline/utils.py so speaker_id.py's embedding step can share the
    same torchcodec-avoidance fix (see module docstring and CODE_REVIEW.md).
    """

    def test_reads_the_given_path_with_expected_soundfile_kwargs(self):
        fake_data = np.zeros((10, 1), dtype="float32")
        fake_soundfile = _fake_soundfile_module(fake_data, sample_rate=16000)

        with patch.dict(sys.modules, {"torch": _fake_torch_module(), "soundfile": fake_soundfile}):
            load_waveform("audio.wav")

        fake_soundfile.read.assert_called_once_with("audio.wav", dtype="float32", always_2d=True)

    def test_returns_a_waveform_sample_rate_dict_transposed_to_channels_first(self):
        # 3 samples, 2 channels - shape (samples, channels), as soundfile returns.
        fake_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype="float32")
        fake_soundfile = _fake_soundfile_module(fake_data, sample_rate=44100)

        with patch.dict(sys.modules, {"torch": _fake_torch_module(), "soundfile": fake_soundfile}):
            result = load_waveform("audio.wav")

        self.assertEqual(set(result.keys()), {"waveform", "sample_rate"})
        self.assertEqual(result["sample_rate"], 44100)
        # load_waveform() transposes to (channels, samples) before returning it.
        np.testing.assert_array_equal(result["waveform"], fake_data.T)


class TestLogAnthropicUsage(unittest.TestCase):
    def test_logs_input_and_output_token_counts(self):
        response = MagicMock()
        response.usage.input_tokens = 1234
        response.usage.output_tokens = 56

        messages = []
        log_anthropic_usage(response, messages.append)

        self.assertTrue(any("1234" in m and "56" in m for m in messages))

    def test_missing_usage_attribute_does_not_raise_or_log(self):
        response = MagicMock(spec=[])  # no .usage attribute at all

        messages = []
        log_anthropic_usage(response, messages.append)  # must not raise

        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
