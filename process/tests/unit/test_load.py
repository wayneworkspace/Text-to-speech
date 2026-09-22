"""
Unit tests for process/pipeline/load/load.py - the final Markdown-writing
stage. This is the stage most directly responsible for the deliverable the
whole project exists to produce (timestamped, per-speaker Markdown), so it
gets its own focused test file.

Run: python -m unittest process.tests.unit.test_load -v
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.load import load as load_module


class TestBuildOutputFilename(unittest.TestCase):
    def test_uses_video_basename_without_extension(self):
        name = load_module.build_output_filename("/some/path/interview_01.mp4")
        self.assertIn("interview_01", name)
        self.assertNotIn(".mp4", name)
        self.assertTrue(name.endswith(" - transcript.md"))

    def test_includes_todays_date_in_dd_mm_yy_format(self):
        name = load_module.build_output_filename("video.mp4")
        self.assertRegex(name, r"^\[\d{2}-\d{2}-\d{2}\] - video - transcript\.md$")


class TestWriteMarkdown(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="load_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_output_dir_and_writes_file(self):
        output_dir = os.path.join(self.tmp_dir, "output")  # does not exist yet
        segments = [{"start": 0, "text": "Hello world", "speaker": "Wayne"}]
        path = load_module.write_markdown(output_dir, "video.mp4", segments)
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("**[00:00] Wayne:** Hello world", content)

    def test_prints_whatever_speaker_label_it_is_given(self):
        # write_markdown does not itself invent "Speaker 1"/"Speaker 2" -
        # that is resolve_speaker_names()'s job (see test_speaker_id.py).
        # It just prints whatever "speaker" value it is given, or omits
        # the name entirely when there is none.
        segments = [
            {"start": 5, "text": "with a name", "speaker": "Speaker 1"},
            {"start": 10, "text": "without a name"},
        ]
        path = load_module.write_markdown(self.tmp_dir, "video.mp4", segments)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("**[00:05] Speaker 1:** with a name", content)
        self.assertIn("**[00:10]** without a name", content)

    def test_low_confidence_segments_get_warning_marker_and_banner(self):
        segments = [{"start": 0, "text": "uncertain text", "low_confidence": True}]
        path = load_module.write_markdown(self.tmp_dir, "video.mp4", segments)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("not confident", content)       # the banner
        self.assertIn("uncertain text ⚠", content)  # the per-line marker

    def test_empty_text_segments_are_skipped(self):
        segments = [{"start": 0, "text": "   "}, {"start": 1, "text": "real text"}]
        path = load_module.write_markdown(self.tmp_dir, "video.mp4", segments)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("[00:00]", content)
        self.assertIn("real text", content)

    def test_topic_heading_inserted_before_correct_segment_index(self):
        segments = [
            {"start": 0, "text": "intro"},
            {"start": 30, "text": "new topic starts here"},
        ]
        topics = [{"index": 1, "title": "Chapter Two"}]
        path = load_module.write_markdown(self.tmp_dir, "video.mp4", segments, topics)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertLess(content.index("## Chapter Two"), content.index("new topic starts here"))
        self.assertLess(content.index("intro"), content.index("## Chapter Two"))


if __name__ == "__main__":
    unittest.main()
