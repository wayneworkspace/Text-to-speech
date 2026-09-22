"""
Unit tests for process/pipeline/batch_state.py - the lock file and
per-video state table that make process/batch.py safe to re-run after a
crash, a killed process, or an overlapping cron trigger. See
CODE_REVIEW.md, sections 7-9, for the design being tested here.

Run: python -m unittest process.tests.unit.test_batch_state -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline import batch_state


def _silent_log(_message):
    pass


def _spawn_and_reap_dead_pid() -> int:
    """A PID guaranteed to belong to no running process right now: spawn a
    trivial child, wait for it to exit, and reap it (so psutil no longer
    sees it as alive) - used to simulate a lock file left behind by a
    crashed batch.py run."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class TestAcquireReleaseLock(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="batch_state_test_")
        self.lock_path = os.path.join(self.tmp_dir, "sub", ".batch.lock")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_acquire_when_no_lock_file(self):
        self.assertTrue(batch_state.acquire_lock(self.lock_path, _silent_log))
        self.assertTrue(os.path.exists(self.lock_path))
        with open(self.lock_path) as f:
            info = json.load(f)
        self.assertEqual(info["pid"], os.getpid())

    def test_second_acquire_blocked_while_first_alive(self):
        self.assertTrue(batch_state.acquire_lock(self.lock_path, _silent_log))
        # Our own PID is alive and the lock is fresh -> a second acquire
        # from "another instance" must back off instead of racing ahead.
        self.assertFalse(batch_state.acquire_lock(self.lock_path, _silent_log))

    def test_stale_lock_dead_pid_is_recovered(self):
        dead_pid = _spawn_and_reap_dead_pid()
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with open(self.lock_path, "w") as f:
            json.dump({"pid": dead_pid, "started_at_epoch": time.time()}, f)

        messages = []
        self.assertTrue(batch_state.acquire_lock(self.lock_path, messages.append))
        self.assertTrue(any("crashed" in m for m in messages), "should log that it recovered a stale lock")
        with open(self.lock_path) as f:
            info = json.load(f)
        self.assertEqual(info["pid"], os.getpid())  # lock now genuinely belongs to us

    def test_stale_lock_by_age_even_if_pid_alive(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        # A PID that IS alive (our own test process), but old enough that
        # the age safety net should kick in regardless of the PID check -
        # this is the "process hung forever without dying" case the PID
        # check alone cannot catch.
        old_started_at = time.time() - (batch_state._MAX_LOCK_AGE_SECONDS + 60)
        with open(self.lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "started_at_epoch": old_started_at}, f)

        self.assertTrue(batch_state.acquire_lock(self.lock_path, _silent_log))

    def test_corrupt_lock_file_is_treated_as_stale(self):
        # Exactly the failure mode this project hit once already with a
        # stale .git/index.lock - must never deadlock the batch runner.
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with open(self.lock_path, "w") as f:
            f.write("{not valid json")
        self.assertTrue(batch_state.acquire_lock(self.lock_path, _silent_log))

    def test_release_lock_removes_file(self):
        batch_state.acquire_lock(self.lock_path, _silent_log)
        batch_state.release_lock(self.lock_path)
        self.assertFalse(os.path.exists(self.lock_path))

    def test_release_lock_is_safe_when_already_gone(self):
        batch_state.release_lock(self.lock_path)  # must not raise


class TestStateLoadSave(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="batch_state_test_")
        self.state_path = os.path.join(self.tmp_dir, "sub", "state.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_missing_file_returns_empty_dict(self):
        self.assertEqual(batch_state.load_state(self.state_path), {})

    def test_save_then_load_roundtrip(self):
        state = {"video1.mp4": {"status": "done", "output_path": "x.md"}}
        batch_state.save_state(self.state_path, state)
        self.assertEqual(batch_state.load_state(self.state_path), state)


class TestRecoverInterrupted(unittest.TestCase):
    def test_running_entries_reset_to_pending_with_bumped_retry(self):
        state = {
            "a.mp4": {"status": "running", "retry_count": 0},
            "b.mp4": {"status": "done"},
            "c.mp4": {"status": "pending"},
        }
        result = batch_state.recover_interrupted(state, _silent_log)
        self.assertEqual(result["a.mp4"]["status"], "pending")
        self.assertEqual(result["a.mp4"]["retry_count"], 1)
        self.assertEqual(result["b.mp4"]["status"], "done")  # untouched
        self.assertEqual(result["c.mp4"]["status"], "pending")  # untouched

    def test_no_running_entries_is_a_no_op(self):
        state = {"a.mp4": {"status": "done"}}
        result = batch_state.recover_interrupted(state, _silent_log)
        self.assertEqual(result, {"a.mp4": {"status": "done"}})


class TestMarkAndShouldProcess(unittest.TestCase):
    def test_should_process_true_for_unknown_video(self):
        self.assertTrue(batch_state.should_process({}, "new.mp4", _silent_log))

    def test_mark_running_then_done(self):
        state = {}
        batch_state.mark_running(state, "a.mp4")
        self.assertEqual(state["a.mp4"]["status"], "running")
        batch_state.mark_done(state, "a.mp4", "/out/a.md")
        self.assertEqual(state["a.mp4"]["status"], "done")
        self.assertEqual(state["a.mp4"]["output_path"], "/out/a.md")
        self.assertNotIn("error", state["a.mp4"])
        self.assertFalse(batch_state.should_process(state, "a.mp4", _silent_log))

    def test_mark_failed_stays_retryable_until_max_retries(self):
        state = {}
        batch_state.mark_running(state, "a.mp4")
        for expected_retry_count in range(1, batch_state.MAX_RETRIES + 1):
            batch_state.mark_failed(state, "a.mp4", "boom")
            self.assertEqual(state["a.mp4"]["retry_count"], expected_retry_count)
            self.assertEqual(state["a.mp4"]["status"], "pending")
            self.assertTrue(batch_state.should_process(state, "a.mp4", _silent_log))

        # One more failure pushes retry_count past MAX_RETRIES...
        batch_state.mark_failed(state, "a.mp4", "boom again")
        self.assertEqual(state["a.mp4"]["retry_count"], batch_state.MAX_RETRIES + 1)
        # ...which should_process() now turns into a permanent "failed", so
        # a broken file is not retried forever, once per cron cycle.
        self.assertFalse(batch_state.should_process(state, "a.mp4", _silent_log))
        self.assertEqual(state["a.mp4"]["status"], "failed")

    def test_should_process_false_for_already_failed(self):
        state = {"a.mp4": {"status": "failed", "retry_count": 99}}
        self.assertFalse(batch_state.should_process(state, "a.mp4", _silent_log))


if __name__ == "__main__":
    unittest.main()
