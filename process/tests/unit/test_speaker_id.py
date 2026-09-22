"""
Unit tests for process/pipeline/enrich/speaker_id.py - the pure,
non-model-dependent pieces of speaker recognition (matching math, name
resolution, profile storage). Computing the embeddings themselves needs
pyannote.audio + a HuggingFace token + real audio, which is out of scope
for this sandboxed suite - see process/tests/README.md.

Run: python -m unittest process.tests.unit.test_speaker_id -v
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from pipeline.enrich import speaker_id


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(v, v), 1.0, places=6)

    def test_opposite_vectors_score_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), -1.0, places=6)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 0.0, places=6)

    def test_known_45_degree_angle(self):
        # [1,0] vs [1,1]: cos(45 deg) = 1/sqrt(2) - hand-computable, unlike
        # arbitrary vectors, so this pins down the exact expected value
        # instead of just checking "some number between 0 and 1".
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 1 / math.sqrt(2), places=6)

    def test_not_affected_by_magnitude(self):
        a = np.array([1.0, 0.0])
        b = np.array([50.0, 0.0])
        self.assertAlmostEqual(speaker_id._cosine_similarity(a, b), 1.0, places=6)


class TestMatchSpeakers(unittest.TestCase):
    def setUp(self):
        # 45-degree pair -> similarity ~0.7071, deliberately chosen so a
        # threshold of 0.50 accepts it and a threshold of 0.75 rejects it.
        self.centroids = {"SPEAKER_00": np.array([1.0, 1.0])}
        self.profiles = [{"name": "Wayne", "embedding": [1.0, 0.0]}]

    def test_match_above_threshold(self):
        result = speaker_id.match_speakers(self.centroids, self.profiles, threshold=0.50)
        self.assertEqual(result, {"SPEAKER_00": "Wayne"})

    def test_no_match_below_threshold(self):
        result = speaker_id.match_speakers(self.centroids, self.profiles, threshold=0.75)
        self.assertEqual(result, {})  # ~0.7071 does not clear 0.75

    def test_picks_the_closer_of_two_profiles(self):
        centroids = {"SPEAKER_00": np.array([1.0, 0.0])}
        profiles = [
            {"name": "Far", "embedding": [0.0, 1.0]},    # similarity 0.0
            {"name": "Close", "embedding": [0.9, 0.1]},  # similarity ~0.994
        ]
        result = speaker_id.match_speakers(centroids, profiles, threshold=0.5)
        self.assertEqual(result, {"SPEAKER_00": "Close"})

    def test_empty_profiles_yields_no_matches(self):
        result = speaker_id.match_speakers(self.centroids, [], threshold=0.1)
        self.assertEqual(result, {})


class TestResolveSpeakerNames(unittest.TestCase):
    def test_matched_speaker_gets_enrolled_name(self):
        segments = [{"speaker": "SPEAKER_00", "text": "hi"}]
        result = speaker_id.resolve_speaker_names(segments, {"SPEAKER_00": "Wayne"})
        self.assertEqual(result[0]["speaker"], "Wayne")

    def test_unmatched_speakers_fall_back_in_order_of_first_appearance(self):
        segments = [
            {"speaker": "SPEAKER_01", "text": "a"},
            {"speaker": "SPEAKER_00", "text": "b"},
            {"speaker": "SPEAKER_01", "text": "c"},
        ]
        result = speaker_id.resolve_speaker_names(segments, {})
        self.assertEqual(result[0]["speaker"], "Speaker 1")  # SPEAKER_01 seen first
        self.assertEqual(result[1]["speaker"], "Speaker 2")  # SPEAKER_00 seen second
        self.assertEqual(result[2]["speaker"], "Speaker 1")  # SPEAKER_01 again -> same label

    def test_mixed_matched_and_unmatched(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "known"},
            {"speaker": "SPEAKER_01", "text": "unknown"},
        ]
        result = speaker_id.resolve_speaker_names(segments, {"SPEAKER_00": "Wayne"})
        self.assertEqual(result[0]["speaker"], "Wayne")
        self.assertEqual(result[1]["speaker"], "Speaker 1")

    def test_segments_without_speaker_field_are_untouched(self):
        segments = [{"text": "no diarization ran"}]
        result = speaker_id.resolve_speaker_names(segments, {})
        self.assertNotIn("speaker", result[0])


class TestProfileStorage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="speaker_id_test_")
        self.path = os.path.join(self.tmp_dir, "sub", "speaker_profiles.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_missing_file_returns_empty_list(self):
        self.assertEqual(speaker_id.load_profiles(self.path), [])

    def test_save_then_load_roundtrip(self):
        profiles = [{"name": "Wayne", "embedding": [1.0, 2.0], "sample_count": 1}]
        speaker_id.save_profiles(self.path, profiles)
        self.assertEqual(speaker_id.load_profiles(self.path), profiles)


if __name__ == "__main__":
    unittest.main()
