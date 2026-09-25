"""
Unit tests for segment_topics() in process/pipeline/enrich/enrich.py - the
optional, Claude-based topic-segmentation step. This module had no direct
unit test coverage before - a real bug in its response parsing (see
test_leading_thinking_block_does_not_break_parsing below) only surfaced
because a sibling module (correct.py) hit the same bug in production first.

No real Anthropic API calls are made - same sys.modules injection
technique as test_correct_transcript.py, since `anthropic` is imported
lazily inside the function under test and is not installed in this sandbox.

Run: python -m unittest tests.unit.test_enrich -v
"""
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.enrich.enrich import segment_topics


def _segments(*texts, speakers=None):
    segs = []
    for i, t in enumerate(texts):
        seg = {"start": float(i), "end": float(i + 1), "text": t}
        if speakers:
            seg["speaker"] = speakers[i]
        segs.append(seg)
    return segs


def _fake_anthropic_module(response_text=None, raise_exc=None, extra_blocks_before=None):
    """
    Same fake as test_correct_transcript.py's helper: fake_module.Anthropic(...)
    returns a client whose .messages.create(...) either returns a response
    whose .content ends in a real text block (type="text"), or raises
    raise_exc. `extra_blocks_before` lets a test prepend non-text blocks
    (e.g. a ThinkingBlock) ahead of the text block.
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


class TestSegmentTopics(unittest.TestCase):
    def test_returns_parsed_topics(self):
        segments = _segments("intro", "chuyen sang chu de moi", "ket thuc")
        response = json.dumps([{"index": 0, "title": "Mở đầu"}, {"index": 1, "title": "Chủ đề chính"}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", _silent_log)

        self.assertEqual(topics, [{"index": 0, "title": "Mở đầu"}, {"index": 1, "title": "Chủ đề chính"}])

    def test_leading_thinking_block_does_not_break_parsing(self):
        # Regression test for the real bug hit in production (see
        # test_correct_transcript.py's version of this test): Claude can put
        # a ThinkingBlock ahead of the text block (extended thinking) -
        # response.content[0].text used to crash on this.
        segments = _segments("intro")
        response = json.dumps([{"index": 0, "title": "Mở đầu"}])
        fake_module, _ = _fake_anthropic_module(
            response_text=response, extra_blocks_before=[MagicMock(type="thinking", text=None)]
        )

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", _silent_log)

        self.assertEqual(topics, [{"index": 0, "title": "Mở đầu"}])

    def test_malformed_json_falls_back_to_empty_list_and_never_raises(self):
        segments = _segments("intro")
        fake_module, _ = _fake_anthropic_module(response_text="not json {{{")

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", messages.append)

        self.assertEqual(topics, [])
        self.assertTrue(any("failed" in m for m in messages))

    def test_api_exception_falls_back_to_empty_list_and_never_raises(self):
        segments = _segments("intro")
        fake_module, _ = _fake_anthropic_module(raise_exc=ConnectionError("network down"))

        messages = []
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", messages.append)

        self.assertEqual(topics, [])
        self.assertTrue(any("failed" in m for m in messages))

    def test_empty_segments_returns_empty_list_without_calling_api(self):
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics([], "key", "model", _silent_log)

        self.assertEqual(topics, [])
        mock_client.messages.create.assert_not_called()

    def test_items_missing_required_fields_are_dropped(self):
        segments = _segments("intro", "sau do")
        response = json.dumps([{"index": 0, "title": "OK"}, {"index": 1}, {"not_a_topic": True}])
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", _silent_log)

        self.assertEqual(topics, [{"index": 0, "title": "OK"}])

    def test_response_wrapped_in_markdown_fence_is_still_parsed(self):
        segments = _segments("intro")
        response = "```json\n" + json.dumps([{"index": 0, "title": "Mở đầu"}]) + "\n```"
        fake_module, _ = _fake_anthropic_module(response_text=response)

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            topics = segment_topics(segments, "key", "model", _silent_log)

        self.assertEqual(topics, [{"index": 0, "title": "Mở đầu"}])

    def test_prompt_includes_speaker_labels_when_present(self):
        segments = _segments("xin chao", "toi la B", speakers=["Speaker 1", "Speaker 2"])
        fake_module, mock_client = _fake_anthropic_module(response_text="[]")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            segment_topics(segments, "key", "model", _silent_log)

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Speaker 1", prompt)
        self.assertIn("Speaker 2", prompt)


if __name__ == "__main__":
    unittest.main()
