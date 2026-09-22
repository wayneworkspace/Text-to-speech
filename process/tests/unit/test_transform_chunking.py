"""
Unit tests for the pure planning logic in
process/pipeline/transform/transform.py (chunk boundary planning) and
process/pipeline/utils.py (timestamp formatting). Nothing here touches
ffmpeg or Whisper - see process/tests/integration/ for that.

Run: python -m unittest process.tests.unit.test_transform_chunking -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.transform import transform as transform_module
from pipeline.utils import format_timestamp


class TestPlanChunks(unittest.TestCase):
    def test_short_video_is_a_single_chunk(self):
        chunks = transform_module.plan_chunks(duration=200, silences=[], target=300, min_len=120, max_len=420)
        self.assertEqual(chunks, [(0.0, 200)])

    def test_video_exactly_at_target_is_a_single_chunk(self):
        chunks = transform_module.plan_chunks(duration=300, silences=[], target=300, min_len=120, max_len=420)
        self.assertEqual(chunks, [(0.0, 300)])

    def test_splits_at_ideal_end_when_no_silence_available(self):
        chunks = transform_module.plan_chunks(duration=1000, silences=[], target=300, min_len=120, max_len=420)
        # No silences to snap to, so every split lands exactly on
        # ideal_end = cursor + target, until the tail is short enough to
        # finish in one last chunk instead of being split again.
        self.assertEqual(chunks[0], (0.0, 300))
        self.assertEqual(chunks[1], (300, 600))
        self.assertEqual(chunks[2], (600, 1000))  # 400s remaining <= max_len (420)
        self.assertEqual(len(chunks), 3)

    def test_prefers_silence_midpoint_closest_to_ideal_end(self):
        # ideal_end = 300; one silence lands right at the ideal point, a
        # further one is outside the [min_len, max_len] window and must
        # be ignored in favor of the closer, in-window one.
        silences = [(298, 302), (500, 502)]
        chunks = transform_module.plan_chunks(duration=1000, silences=silences, target=300, min_len=120, max_len=420)
        self.assertEqual(chunks[0], (0.0, 300.0))  # midpoint of (298, 302)

    def test_ignores_silence_outside_the_allowed_window(self):
        # min_len=120, max_len=420 -> window is [cursor+120, cursor+420].
        # A silence at t=50 (midpoint) is before the window and must be
        # skipped in favor of falling back to ideal_end.
        silences = [(48, 52)]
        chunks = transform_module.plan_chunks(duration=1000, silences=silences, target=300, min_len=120, max_len=420)
        self.assertEqual(chunks[0], (0.0, 300))  # fell back to ideal_end

    def test_all_chunk_boundaries_cover_the_full_duration_without_gaps(self):
        silences = [(i, i + 1) for i in range(50, 950, 77)]  # scattered silences
        chunks = transform_module.plan_chunks(duration=1000, silences=silences, target=300, min_len=120, max_len=420)
        self.assertAlmostEqual(chunks[0][0], 0.0)
        self.assertAlmostEqual(chunks[-1][1], 1000)
        for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(end, next_start)  # no gaps, no overlap


class TestFormatTimestamp(unittest.TestCase):
    def test_seconds_and_minutes_only_under_an_hour(self):
        self.assertEqual(format_timestamp(65), "01:05")

    def test_includes_hours_when_an_hour_or_more(self):
        self.assertEqual(format_timestamp(3661), "01:01:01")

    def test_zero_is_00_00(self):
        self.assertEqual(format_timestamp(0), "00:00")

    def test_negative_clamped_to_zero(self):
        self.assertEqual(format_timestamp(-5), "00:00")


if __name__ == "__main__":
    unittest.main()
