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


def _fake_anthropic_module(response_text=None, raise_exc=None, extra_blocks_before=None):
    """
    A minimal stand-in for the `anthropic` package: fake_module.Anthropic(...)
    returns a client whose .messages.create(...) either returns a response
    whose .content ends in a real text block (type="text", matching the
    real SDK), or raises raise_exc. `extra_blocks_before` lets a test
    prepend non-text blocks (e.g. a ThinkingBlock) ahead of the text block,
    the way extended thinking legitimately does.
    """
    fake_module = types.ModuleType("anthropic")
    mock_client = MagicMock()
    if raise_exc is not None:
        mock_client.messages.create.side_effect = raise_exc
    else:
        mock_response = MagicMock()
        blocks = list(extra_blocks_before or [])
        blocks.append(MagicMock(type="text", text=response_text))
        mock_response.content = blocks
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

    def test_leading_thinking_block_does_not_break_parsing(self):
        # Regression test for a real bug hit running this project live:
        # Claude can put a ThinkingBlock ahead of the text block (extended
        # thinking) - response.content[0].text used to crash on this,
        # silently falling back and skipping every correction.
        segments = _segments("xin chao")
        response = json.dumps([{"index": 0, "text": "xin chào"}])
        fake_module, _ = _fake_anthropic_module(
            response_text=response, extra_blocks_before=[MagicMock(type="thinking", text=None)]
        )

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        self.assertEqual(result[0]["text"], "xin chào")

    def test_prompt_omits_domain_vocabulary_section_when_not_set(self):
        segments = _segments("noi dung")
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log)

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("Known domain vocabulary", prompt)


class TestCorrectTranscriptErrorsBatching(unittest.TestCase):
    """
    Regression tests for the batch_size split added after a real failure
    running this project on a ~2 hour video: one call covering the whole
    ~1500-line transcript spent its entire output budget on extended
    thinking and returned no text block at all, so correction silently
    never applied for that entire run (see CODE_REVIEW.md). Splitting into
    batches keeps each call's input/output small regardless of video
    length, and a failed batch must not cost any other batch's corrections.
    """

    def test_segments_within_one_batch_makes_a_single_call(self):
        segments = _segments("mot", "hai", "ba")
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log, batch_size=150)

        self.assertEqual(mock_client.messages.create.call_count, 1)

    def test_segments_exceeding_batch_size_makes_multiple_calls(self):
        segments = _segments(*[f"line {i}" for i in range(10)])
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log, batch_size=4)

        # 10 lines / batch_size 4 -> batches of 4, 4, 2 = 3 calls.
        self.assertEqual(mock_client.messages.create.call_count, 3)

    def test_second_batchs_line_numbers_use_the_global_index_not_zero(self):
        segments = _segments(*[f"line {i}" for i in range(6)])
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log, batch_size=3)

        first_prompt = mock_client.messages.create.call_args_list[0].kwargs["messages"][0]["content"]
        second_prompt = mock_client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
        self.assertIn("[0] line 0", first_prompt)
        self.assertIn("[3] line 3", second_prompt)  # global index, not restarting at [0]

    def test_corrections_from_every_batch_are_applied(self):
        segments = _segments(*[f"line {i}" for i in range(6)])
        responses = [
            json.dumps([{"index": 0, "text": "FIXED 0"}]),
            json.dumps([{"index": 3, "text": "FIXED 3"}]),
        ]
        fake_module = types.ModuleType("anthropic")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(type="text", text=r)]) for r in responses
        ]
        fake_module.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(segments, "v.mp4", None, "key", "model", _silent_log, batch_size=3)

        self.assertEqual(result[0]["text"], "FIXED 0")
        self.assertEqual(result[3]["text"], "FIXED 3")
        self.assertEqual(result[1]["text"], "line 1")  # untouched

    def test_one_failed_batch_does_not_lose_another_batchs_corrections(self):
        segments = _segments(*[f"line {i}" for i in range(6)])
        fake_module = types.ModuleType("anthropic")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            ConnectionError("network down"),  # first batch fails outright
            MagicMock(content=[MagicMock(type="text", text=json.dumps([{"index": 3, "text": "FIXED 3"}]))]),
        ]
        fake_module.Anthropic = MagicMock(return_value=mock_client)

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(
                segments, "v.mp4", None, "key", "model", messages.append, batch_size=3
            )

        self.assertEqual(result[0]["text"], "line 0")  # first batch: left alone, not crashed
        self.assertEqual(result[3]["text"], "FIXED 3")  # second batch: still applied
        self.assertTrue(any("lines 0-2" in m and "failed" in m for m in messages))

    def test_out_of_range_index_in_one_batch_does_not_affect_other_batches(self):
        segments = _segments(*[f"line {i}" for i in range(6)])
        fake_module = types.ModuleType("anthropic")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(type="text", text=json.dumps([{"index": 99, "text": "bogus"}]))]),
            MagicMock(content=[MagicMock(type="text", text=json.dumps([{"index": 3, "text": "FIXED 3"}]))]),
        ]
        fake_module.Anthropic = MagicMock(return_value=mock_client)

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(
                segments, "v.mp4", None, "key", "model", messages.append, batch_size=3
            )

        self.assertEqual(result[0]["text"], "line 0")
        self.assertEqual(result[3]["text"], "FIXED 3")
        self.assertTrue(any("out-of-range" in m for m in messages))

    def test_empty_text_response_like_the_real_bug_is_handled_gracefully(self):
        # The actual failure mode hit in production: extended thinking used
        # the whole output budget, so response.content had no text block at
        # all - extract_text_from_anthropic_response() returns "", and
        # json.loads("") raises "Expecting value: line 1 column 1 (char 0)".
        segments = _segments(*[f"line {i}" for i in range(6)])
        fake_module = types.ModuleType("anthropic")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(type="thinking", text=None)]),  # no text block at all
            MagicMock(content=[MagicMock(type="text", text=json.dumps([{"index": 3, "text": "FIXED 3"}]))]),
        ]
        fake_module.Anthropic = MagicMock(return_value=mock_client)

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = correct_transcript_errors(
                segments, "v.mp4", None, "key", "model", messages.append, batch_size=3
            )

        self.assertEqual(result[0]["text"], "line 0")  # first batch failed, left alone
        self.assertEqual(result[3]["text"], "FIXED 3")  # second batch unaffected
        self.assertTrue(any("lines 0-2" in m and "failed" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
