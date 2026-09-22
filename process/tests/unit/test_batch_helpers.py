"""
Unit tests for process/batch.py's own helper functions - find_video_files,
_is_file_stable, _clean_orphaned_temp_dirs, _make_file_logger. These are
plain filesystem logic, independent of pipeline/batch_state.py (already
covered in test_batch_state.py) and of the real pipeline run itself.

Run: python -m unittest tests.unit.test_batch_helpers -v
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import batch


class TestFindVideoFiles(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="find_videos_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _touch(self, *parts):
        path = os.path.join(self.tmp_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_finds_video_files_recursively(self):
        self._touch("a.mp4")
        self._touch("sub", "b.mov")
        self._touch("sub", "deeper", "c.mkv")
        found = batch.find_video_files(self.tmp_dir)
        names = {os.path.relpath(p, self.tmp_dir) for p in found}
        self.assertEqual(names, {"a.mp4", os.path.join("sub", "b.mov"), os.path.join("sub", "deeper", "c.mkv")})

    def test_filters_out_non_video_extensions(self):
        self._touch("real.mp4")
        self._touch("notes.txt")
        self._touch("subtitle.srt")
        self._touch("thumbnail.jpg")
        found = batch.find_video_files(self.tmp_dir)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].endswith("real.mp4"))

    def test_extension_match_is_case_insensitive(self):
        self._touch("Video.MP4")
        found = batch.find_video_files(self.tmp_dir)
        self.assertEqual(len(found), 1)

    def test_accepts_all_documented_extensions(self):
        for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
            self._touch(f"clip{ext}")
        found = batch.find_video_files(self.tmp_dir)
        self.assertEqual(len(found), 5)

    def test_results_are_sorted(self):
        self._touch("charlie.mp4")
        self._touch("alpha.mp4")
        self._touch("bravo.mp4")
        found = batch.find_video_files(self.tmp_dir)
        self.assertEqual(found, sorted(found))

    def test_empty_directory_returns_empty_list(self):
        self.assertEqual(batch.find_video_files(self.tmp_dir), [])


class TestIsFileStable(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="is_stable_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stable_nonempty_file_returns_true(self):
        path = os.path.join(self.tmp_dir, "video.mp4")
        with open(path, "wb") as f:
            f.write(b"x" * 100)
        self.assertTrue(batch._is_file_stable(path, wait_seconds=0.01))

    def test_missing_file_returns_false(self):
        path = os.path.join(self.tmp_dir, "does_not_exist.mp4")
        self.assertFalse(batch._is_file_stable(path, wait_seconds=0.01))

    def test_stable_but_empty_file_returns_false(self):
        # Still being created / zero bytes is never "safe to process", even
        # if its size happens not to change during the check window.
        path = os.path.join(self.tmp_dir, "empty.mp4")
        open(path, "w").close()
        self.assertFalse(batch._is_file_stable(path, wait_seconds=0.01))

    def test_growing_file_returns_false(self):
        # Simulate a file whose size changes between the two reads (still
        # being copied into INPUT_DIR) without relying on real timing.
        path = os.path.join(self.tmp_dir, "copying.mp4")
        open(path, "w").close()
        with patch("batch.time.sleep"), patch("batch.os.path.getsize", side_effect=[100, 250]):
            self.assertFalse(batch._is_file_stable(path))


class TestCleanOrphanedTempDirs(unittest.TestCase):
    def setUp(self):
        self.fake_tempdir = tempfile.mkdtemp(prefix="fake_system_temp_")

    def tearDown(self):
        shutil.rmtree(self.fake_tempdir, ignore_errors=True)

    def _mkdir(self, name):
        path = os.path.join(self.fake_tempdir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_removes_video2text_and_enroll_prefixed_dirs(self):
        leftover1 = self._mkdir("video2text_abc123")
        leftover2 = self._mkdir("enroll_xyz789")
        with patch("batch.tempfile.gettempdir", return_value=self.fake_tempdir):
            batch._clean_orphaned_temp_dirs(lambda m: None)
        self.assertFalse(os.path.exists(leftover1))
        self.assertFalse(os.path.exists(leftover2))

    def test_leaves_unrelated_directories_alone(self):
        unrelated = self._mkdir("some_other_app_cache")
        with patch("batch.tempfile.gettempdir", return_value=self.fake_tempdir):
            batch._clean_orphaned_temp_dirs(lambda m: None)
        self.assertTrue(os.path.exists(unrelated))

    def test_logs_only_when_something_was_removed(self):
        messages = []
        with patch("batch.tempfile.gettempdir", return_value=self.fake_tempdir):
            batch._clean_orphaned_temp_dirs(messages.append)
        self.assertEqual(messages, [])  # nothing to clean -> silent

        self._mkdir("video2text_leftover")
        messages2 = []
        with patch("batch.tempfile.gettempdir", return_value=self.fake_tempdir):
            batch._clean_orphaned_temp_dirs(messages2.append)
        self.assertEqual(len(messages2), 1)
        self.assertIn("1", messages2[0])


class TestMakeFileLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="file_logger_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_message_to_all_given_files(self):
        path1 = os.path.join(self.tmp_dir, "a.log")
        path2 = os.path.join(self.tmp_dir, "b.log")
        log = batch._make_file_logger(path1, path2)
        log("hello world")

        for path in (path1, path2):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello world", content)
            self.assertTrue(content.strip().startswith("["))  # timestamp prefix

    def test_creates_missing_parent_directories(self):
        path = os.path.join(self.tmp_dir, "nested", "deep", "run.log")
        log = batch._make_file_logger(path)
        log("first line")
        self.assertTrue(os.path.exists(path))

    def test_appends_across_multiple_calls(self):
        path = os.path.join(self.tmp_dir, "run.log")
        log = batch._make_file_logger(path)
        log("line one")
        log("line two")
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l]
        self.assertEqual(len(lines), 2)
        self.assertIn("line one", lines[0])
        self.assertIn("line two", lines[1])


if __name__ == "__main__":
    unittest.main()
