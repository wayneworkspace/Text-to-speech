"""
Unit tests for correct_transcript_errors() in
process/pipeline/enrich/correct.py - the optional, Claude-based
transcription-error fixer.

No real Anthropic API calls are made: the `anthropic` package is not even
installed in this sandbox (it is imported lazily inside the function under
test, same pattern as pipeline/enrich/enrich.py's segment_topics()), so
these tests inject a fake module into sys.modules before calling the
function - the standard way to test code that imports an optional
dependency you don't have installed.

Run: python -m unittest tests.unit.test_correct_transcript -v
"""
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.enrich.correct import correct_transcript_errors


def _segments(*texts):
    return [
        {"start": float(i), "end": float(i + 1), "text": t, "speaker": "Wayne", "low_confidence": False}
        for i, t in enumerate(texts)
    ]


def _fake_anthropic_module(response_text=None, raise_exc=None):
    """
    A minimal stand-in for the `anthropic` package: fake_module.Anthropic(...)
    returns a client whose .messages.create(...) either returns a response
    object with .content[0].text == response_text, or raises raise_exc.
    """
    fake_module = types.ModuleType("anthropic")
    mock_client = MagicMock()
    if raise_exc is not None:
        mock_client.messages.create.side_effect = raise_exc
    else:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]
        mock_client.messages.create.return_value = mock_response
    fake_module.Anthropic = MagicMock(return_value=mock_client)
    return fake_module, mock_client


def _silent_log(_message):
    pass


class TestCorrectTranscriptErrors(unittest.TestCase):
    def test_applies_corrections_to_matching_indices_only(self):
        segments = _segments("xin chao", "cai nay dung roi", "tam biet")
        response = json.dumps([{"index": 0, "text": "xin chào"}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result[0]["text"], "xin chào")
        self.assertEqual(result[1]["text"], "cai nay dung roi")  # untouched
        self.assertEqual(result[2]["text"], "tam biet")  # untouched

    def test_only_text_field_changes(self):
        segments = _segments("sai roi")
        segments[0]["speaker"] = "Wayne"
        segments[0]["low_confidence"] = True
        response = json.dumps([{"index": 0, "text": "sửa rồi"}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result[0]["text"], "sửa rồi")
        self.assertEqual(result[0]["speaker"], "Wayne")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 1.0)
        self.assertTrue(result[0]["low_confidence"])

    def test_out_of_range_index_falls_back_to_original(self):
        segments = _segments("mot", "hai")
        response = json.dumps([{"index": 5, "text": "something"}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", messages.append)

        self.assertEqual(result[0]["text"], "mot")
        self.assertEqual(result[1]["text"], "hai")
        self.assertTrue(any("out-of-range" in m for m in messages))

    def test_negative_index_falls_back_to_original(self):
        segments = _segments("mot")
        response = json.dumps([{"index": -1, "text": "something"}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result[0]["text"], "mot")

    def test_malformed_json_response_falls_back_to_original(self):
        segments = _segments("mot", "hai")
        fake_module, _ = _fake_anthropic_module(response_text="this is not json at all {{{")

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", messages.append)

        self.assertEqual([s["text"] for s in result], ["mot", "hai"])
        self.assertTrue(any("failed" in m for m in messages))

    def test_api_exception_falls_back_to_original_and_never_raises(self):
        segments = _segments("mot", "hai")
        fake_module, _ = _fake_anthropic_module(raise_exc=ConnectionError("network down"))

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", messages.append)

        self.assertEqual([s["text"] for s in result], ["mot", "hai"])
        self.assertTrue(any("failed" in m for m in messages))

    def test_empty_segments_returns_immediately_without_calling_api(self):
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors([], "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result, [])
        mock_client.messages.create.assert_not_called()

    def test_empty_valid_response_leaves_segments_unchanged(self):
        segments = _segments("mot dung roi")
        fake_module, _ = _fake_anthropic_module(response_text="[]")

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", messages.append)

        self.assertEqual(result[0]["text"], "mot dung roi")
        self.assertTrue(any("No likely transcription errors" in m for m in messages))

    def test_identical_replacement_text_does_not_count_as_changed(self):
        segments = _segments("cau dung roi")
        response = json.dumps([{"index": 0, "text": "cau dung roi"}])  # same text, wasteful but valid
        fake_module, _ = _fake_anthropic_module(response_text=response)

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", messages.append)

        self.assertTrue(any("Corrected 0 likely" in m for m in messages))

    def test_response_wrapped_in_markdown_fence_is_still_parsed(self):
        segments = _segments("xin chao")
        response = "```json\n" + json.dumps([{"index": 0, "text": "xin chào"}]) + "\n```"
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result[0]["text"], "xin chào")

    def test_prompt_includes_video_filename(self):
        segments = _segments("noi dung")
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(
                segments, "/some/path/intro_SuongTrangMienQueNgoai.mp4", None, "key", "model", _silent_log
            )

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("intro_SuongTrangMienQueNgoai.mp4", prompt)

    def test_prompt_includes_domain_vocabulary_when_set(self):
        segments = _segments("noi dung")
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(
                segments, "v.mp4", "Sương Trắng Miền Quê Ngoại, luân phiên", "key", "model", _silent_log
            )

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Sương Trắng Miền Quê Ngoại", prompt)

    def test_prompt_omits_domain_vocabulary_section_when_not_set(self):
        segments = _segments("noi dung")
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("Known domain vocabulary", prompt)


if __name__ == "__main__":
    unittest.main()
