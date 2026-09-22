"""
Unit tests for assign_speakers() in process/pipeline/enrich/diarize.py -
the pure interval-overlap logic that attaches a speaker label to each
Whisper segment. diarize_audio() itself (the real pyannote.audio call) is
not covered here - see process/tests/README.md for why.

Run: python -m unittest tests.unit.test_diarize -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.enrich.diarize import assign_speakers


def _seg(start, end):
    return {"start": start, "end": end, "text": "..."}


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


if __name__ == "__main__":
    unittest.main()
