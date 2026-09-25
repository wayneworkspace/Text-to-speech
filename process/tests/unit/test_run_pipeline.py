"""
Unit tests for main.py's run_pipeline() - the top-level orchestrator that
calls every pipeline stage in order. Every imported stage function is
mocked (see STAGE_NAMES below), so this suite needs no real Whisper /
pyannote / Claude call and runs fully offline.

It exists to lock in the orchestration contract that process/batch.py's
mark_failed()/mark_done() logic depends on: stage order, the two API-key
gates (HUGGINGFACE_TOKEN, ANTHROPIC_API_KEY), temp-dir cleanup (always,
even when a stage raises), and the return value.

Run: python -m unittest tests.unit.test_run_pipeline -v
"""
import contextlib
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import main

# Every function run_pipeline() calls, in the order it appears in main.py.
# All of these are imported by name into main's namespace (see main.py's
# imports), so each is patched as "main.<name>", not at its origin module.
STAGE_NAMES = [
    "check_ffmpeg_available",
    "probe_media_info",
    "extract_audio",
    "load_whisper_model",
    "transform_to_segments",
    "correct_transcript_errors",
    "diarize_audio",
    "assign_speakers",
    "load_profiles",
    "compute_speaker_centroids",
    "match_speakers",
    "resolve_speaker_names",
    "segment_topics",
    "write_markdown",
]

# Harmless default return value for each stage, used unless a test overrides it.
_DEFAULT_RETURNS = {
    "check_ffmpeg_available": None,
    "probe_media_info": {"duration": 42.0},
    "extract_audio": None,
    "load_whisper_model": mock.sentinel.model,
    "transform_to_segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
    "correct_transcript_errors": [{"start": 0.0, "end": 1.0, "text": "hi (corrected)"}],
    "diarize_audio": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    "assign_speakers": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}],
    "load_profiles": {},
    "compute_speaker_centroids": {},
    "match_speakers": {},
    "resolve_speaker_names": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Speaker 1"}],
    "segment_topics": [{"index": 0, "title": "Topic"}],
    "write_markdown": "/fake/output/video.md",
}


def _noop_log(_msg):
    pass


@contextlib.contextmanager
def _patched_stages(order=None, overrides=None):
    """
    Patch every stage function in main's namespace with a Mock returning
    its default (see _DEFAULT_RETURNS), optionally replaced per-name via
    `overrides` (e.g. {"transform_to_segments": RuntimeError("boom")} to
    make that stage raise instead of return). If `order` is a list, each
    mock appends its own name to it when called, so tests can assert on
    call order. Yields a dict of {name: Mock}.
    """
    overrides = overrides or {}

    def _make_side_effect(name):
        value = overrides.get(name, _DEFAULT_RETURNS[name])

        def _fn(*args, **kwargs):
            if order is not None:
                order.append(name)
            if isinstance(value, BaseException):
                raise value
            return value

        return _fn

    with contextlib.ExitStack() as stack:
        mocks = {}
        for name in STAGE_NAMES:
            m = stack.enter_context(mock.patch(f"main.{name}"))
            m.side_effect = _make_side_effect(name)
            mocks[name] = m
        yield mocks


# Captured before any test patches "main.tempfile.mkdtemp" - main.tempfile
# *is* this same tempfile module object (plain `import tempfile`), so
# patching that attribute patches it here too; calling tempfile.mkdtemp()
# from inside the fake below would recurse into the mock itself.
_REAL_MKDTEMP = tempfile.mkdtemp


def _tmpdir_tracker():
    """
    Returns (created, fake_mkdtemp): fake_mkdtemp calls the real
    tempfile.mkdtemp (so run_pipeline's cleanup logic runs against a real
    directory) while recording every path it creates, so a test can assert
    the directory was actually removed (or actually kept) afterward.
    """
    created = []

    def _fake_mkdtemp(prefix=None):
        d = _REAL_MKDTEMP(prefix=prefix)
        created.append(d)
        return d

    return created, _fake_mkdtemp


class TestStageCallOrder(unittest.TestCase):
    def test_full_order_with_both_api_keys_set(self):
        order = []
        with mock.patch.dict(main.CONFIG, {"huggingface_token": "hf_token", "anthropic_api_key": "sk-ant"}):
            with _patched_stages(order=order):
                main.run_pipeline("video.mp4", _noop_log)

        self.assertEqual(
            order,
            [
                "check_ffmpeg_available",
                "probe_media_info",
                "extract_audio",
                "load_whisper_model",
                "transform_to_segments",
                "correct_transcript_errors",
                "diarize_audio",
                "assign_speakers",
                "load_profiles",
                "compute_speaker_centroids",
                "match_speakers",
                "resolve_speaker_names",
                "segment_topics",
                "write_markdown",
            ],
        )

    def test_return_value_is_write_markdowns_return_value(self):
        with _patched_stages() as mocks:
            result = main.run_pipeline("video.mp4", _noop_log)

        self.assertEqual(result, _DEFAULT_RETURNS["write_markdown"])


class TestHuggingfaceTokenGating(unittest.TestCase):
    _DIARIZATION_STAGES = (
        "diarize_audio",
        "assign_speakers",
        "load_profiles",
        "compute_speaker_centroids",
        "match_speakers",
        "resolve_speaker_names",
    )

    def test_diarization_skipped_when_no_token(self):
        with mock.patch.dict(main.CONFIG, {"huggingface_token": None}):
            with _patched_stages() as mocks:
                main.run_pipeline("video.mp4", _noop_log)

        for name in self._DIARIZATION_STAGES:
            mocks[name].assert_not_called()

    def test_diarization_run_when_token_set(self):
        with mock.patch.dict(main.CONFIG, {"huggingface_token": "hf_token"}):
            with _patched_stages() as mocks:
                main.run_pipeline("video.mp4", _noop_log)

        for name in self._DIARIZATION_STAGES:
            mocks[name].assert_called_once()

    def test_diarization_failure_is_caught_and_pipeline_continues(self):
        # Regression test: a real pyannote/huggingface_hub version mismatch
        # used to crash the whole pipeline here, throwing away the
        # transcription Whisper already did (see CODE_REVIEW.md). Confirms
        # the try/except added around this block in main.py actually
        # protects the rest of the run - later stages still run and
        # run_pipeline() returns normally instead of raising.
        with mock.patch.dict(main.CONFIG, {"huggingface_token": "hf_token"}):
            with _patched_stages(overrides={"diarize_audio": RuntimeError("pyannote boom")}) as mocks:
                result = main.run_pipeline("video.mp4", _noop_log)

        mocks["assign_speakers"].assert_not_called()
        mocks["load_profiles"].assert_not_called()
        mocks["write_markdown"].assert_called_once()
        self.assertEqual(result, _DEFAULT_RETURNS["write_markdown"])


class TestAnthropicApiKeyGating(unittest.TestCase):
    def test_correction_and_topics_skipped_when_no_key(self):
        with mock.patch.dict(main.CONFIG, {"anthropic_api_key": None}):
            with _patched_stages() as mocks:
                main.run_pipeline("video.mp4", _noop_log)

        mocks["correct_transcript_errors"].assert_not_called()
        mocks["segment_topics"].assert_not_called()
        # write_markdown must still get an (empty) topics list, not None.
        write_markdown_args = mocks["write_markdown"].call_args.args
        self.assertEqual(write_markdown_args[3], [])

    def test_correction_and_topics_run_when_key_set(self):
        with mock.patch.dict(main.CONFIG, {"anthropic_api_key": "sk-ant"}):
            with _patched_stages() as mocks:
                main.run_pipeline("video.mp4", _noop_log)

        mocks["correct_transcript_errors"].assert_called_once()
        mocks["segment_topics"].assert_called_once()


class TestTempDirCleanup(unittest.TestCase):
    def test_tmp_dir_removed_on_success(self):
        created, fake_mkdtemp = _tmpdir_tracker()
        with mock.patch.dict(main.CONFIG, {"keep_temp_files": False}):
            with mock.patch("main.tempfile.mkdtemp", side_effect=fake_mkdtemp):
                with _patched_stages():
                    main.run_pipeline("video.mp4", _noop_log)

        self.assertEqual(len(created), 1)
        self.assertFalse(os.path.exists(created[0]))

    def test_tmp_dir_kept_when_keep_temp_files_true(self):
        created, fake_mkdtemp = _tmpdir_tracker()
        try:
            with mock.patch.dict(main.CONFIG, {"keep_temp_files": True}):
                with mock.patch("main.tempfile.mkdtemp", side_effect=fake_mkdtemp):
                    with _patched_stages():
                        main.run_pipeline("video.mp4", _noop_log)

            self.assertEqual(len(created), 1)
            self.assertTrue(os.path.exists(created[0]))
        finally:
            # Clean up for real - KEEP_TEMP_FILES is the one path where
            # run_pipeline itself intentionally does not do this.
            for d in created:
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    def test_tmp_dir_removed_and_exception_still_propagates_on_stage_failure(self):
        # This is the contract batch.py's mark_failed()/mark_done() logic
        # depends on: a failing stage must not leave a temp dir behind, and
        # must not be swallowed - the exception has to reach the caller.
        created, fake_mkdtemp = _tmpdir_tracker()
        boom = RuntimeError("transcription blew up")
        with mock.patch.dict(main.CONFIG, {"keep_temp_files": False}):
            with mock.patch("main.tempfile.mkdtemp", side_effect=fake_mkdtemp):
                with _patched_stages(overrides={"transform_to_segments": boom}) as mocks:
                    with self.assertRaises(RuntimeError) as ctx:
                        main.run_pipeline("video.mp4", _noop_log)

        self.assertIs(ctx.exception, boom)
        self.assertEqual(len(created), 1)
        self.assertFalse(os.path.exists(created[0]))
        # Nothing downstream of the failed stage should have run.
        mocks["correct_transcript_errors"].assert_not_called()
        mocks["write_markdown"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
