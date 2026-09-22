"""
Unit tests for process/config.py - the .env parsing helpers and the
fail-fast chunk-settings validator.

Run: python -m unittest process.tests.unit.test_config -v
(see process/tests/README.md for other ways to run the suite)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config


class TestEnvHelpers(unittest.TestCase):
    """
    config._get_*() read from os.environ at call time, so these tests set
    and restore individual environment variables around each call instead
    of touching the already-loaded config.CONFIG dict.
    """

    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_get_str_uses_default_when_unset(self):
        os.environ.pop("TEST_KEY", None)
        self.assertEqual(config._get_str("TEST_KEY", "fallback"), "fallback")

    def test_get_str_uses_default_when_empty(self):
        os.environ["TEST_KEY"] = ""
        self.assertEqual(config._get_str("TEST_KEY", "fallback"), "fallback")

    def test_get_str_reads_set_value(self):
        os.environ["TEST_KEY"] = "medium"
        self.assertEqual(config._get_str("TEST_KEY", "fallback"), "medium")

    def test_get_optional_str_empty_means_none(self):
        # LANGUAGE="" in .env means "auto-detect" (None), not the literal
        # string "" - this is the one place _get_str's semantics would be
        # wrong, which is why _get_optional_str exists as a separate helper.
        os.environ["TEST_KEY"] = ""
        self.assertIsNone(config._get_optional_str("TEST_KEY", "vi"))

    def test_get_optional_str_unset_uses_default(self):
        os.environ.pop("TEST_KEY", None)
        self.assertEqual(config._get_optional_str("TEST_KEY", "vi"), "vi")

    def test_get_int_parses_and_defaults(self):
        os.environ["TEST_KEY"] = "42"
        self.assertEqual(config._get_int("TEST_KEY", 7), 42)
        os.environ.pop("TEST_KEY", None)
        self.assertEqual(config._get_int("TEST_KEY", 7), 7)

    def test_get_float_parses_and_defaults(self):
        os.environ["TEST_KEY"] = "-30.5"
        self.assertEqual(config._get_float("TEST_KEY", 0.0), -30.5)

    def test_get_bool_truthy_values(self):
        for value in ("1", "true", "True", "yes", "on", "  TRUE  "):
            os.environ["TEST_KEY"] = value
            self.assertTrue(config._get_bool("TEST_KEY", False), f"{value!r} should be truthy")

    def test_get_bool_falsy_and_default(self):
        os.environ["TEST_KEY"] = "0"
        self.assertFalse(config._get_bool("TEST_KEY", True))
        os.environ.pop("TEST_KEY", None)
        self.assertTrue(config._get_bool("TEST_KEY", True))


class TestValidateChunkSettings(unittest.TestCase):
    """
    _validate_chunk_settings() is what stops a bad .env (e.g. CHUNK_MIN_SECONDS
    > CHUNK_MAX_SECONDS) from silently reaching plan_chunks() and producing
    nonsensical chunk boundaries later - see CODE_REVIEW.md, section 2.
    """

    def _cfg(self, min_s, target_s, max_s):
        return {"chunk_min_seconds": min_s, "chunk_target_seconds": target_s, "chunk_max_seconds": max_s}

    def test_accepts_valid_ordering(self):
        config._validate_chunk_settings(self._cfg(120, 300, 420))  # must not raise

    def test_accepts_all_equal(self):
        config._validate_chunk_settings(self._cfg(300, 300, 300))  # boundary case, must not raise

    def test_rejects_min_greater_than_target(self):
        with self.assertRaises(ValueError):
            config._validate_chunk_settings(self._cfg(400, 300, 420))

    def test_rejects_target_greater_than_max(self):
        with self.assertRaises(ValueError):
            config._validate_chunk_settings(self._cfg(120, 500, 420))

    def test_rejects_zero_min(self):
        with self.assertRaises(ValueError):
            config._validate_chunk_settings(self._cfg(0, 300, 420))

    def test_rejects_negative_min(self):
        with self.assertRaises(ValueError):
            config._validate_chunk_settings(self._cfg(-10, 300, 420))


class TestRealConfigLoaded(unittest.TestCase):
    """Sanity checks against the actual CONFIG this project loads at import time."""

    def test_config_has_all_expected_keys(self):
        expected = {
            "model_size", "language", "chunk_target_seconds", "chunk_min_seconds",
            "chunk_max_seconds", "silence_noise_db", "silence_min_duration",
            "keep_temp_files", "domain_vocabulary", "confidence_threshold",
            "huggingface_token", "anthropic_api_key", "anthropic_model",
            "speaker_profiles_path", "speaker_match_threshold",
            "auto_update_speaker_profiles", "extract_chunk_seconds",
            "extract_max_workers", "input_dir", "state_path", "lock_path", "log_dir",
        }
        self.assertTrue(expected.issubset(config.CONFIG.keys()))

    def test_real_config_passes_its_own_validation(self):
        config._validate_chunk_settings(config.CONFIG)  # must not raise


if __name__ == "__main__":
    unittest.main()
