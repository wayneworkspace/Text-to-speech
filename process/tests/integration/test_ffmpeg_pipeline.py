"""
Integration tests for process/pipeline/extract/extract.py and the ffmpeg-
calling parts of process/pipeline/transform/transform.py, run against real
ffmpeg/ffprobe with small synthetic media generated on the fly (ffmpeg's
"lavfi" virtual input - no fixture files committed to the repo, nothing to
download).

These need real ffmpeg/ffprobe on PATH; the whole module skips cleanly
(not a failure) if they are missing. Everything Whisper/pyannote-dependent
is out of scope here - see process/tests/README.md.

Run: python -m unittest process.tests.integration.test_ffmpeg_pipeline -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.utils import FFMPEG_BIN, FFPROBE_BIN, check_ffmpeg_available
from pipeline.extract import extract as extract_module
from pipeline.transform import transform as transform_module

try:
    check_ffmpeg_available()
    _FFMPEG_AVAILABLE = True
except RuntimeError:
    _FFMPEG_AVAILABLE = False


def _make_test_video(path: str, seconds: float, with_audio: bool = True) -> None:
    """Generate a tiny synthetic video with ffmpeg's lavfi virtual input - a
    color video track plus (optionally) a sine-wave audio track. No fixture
    files needed, so this suite has nothing to commit or download."""
    cmd = [FFMPEG_BIN, "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-t", str(seconds)]
    cmd += ["-c:a", "aac"] if with_audio else ["-an"]
    cmd += [path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate test video: {result.stderr[-2000:]}")


def _wav_duration(path: str) -> float:
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg/ffprobe not found on PATH")
class TestProbeMediaInfo(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="probe_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_reports_correct_duration_and_has_audio(self):
        path = os.path.join(self.tmp_dir, "clip.mp4")
        _make_test_video(path, seconds=3, with_audio=True)
        info = extract_module.probe_media_info(path)
        self.assertAlmostEqual(info["duration"], 3, delta=0.5)

    def test_raises_when_no_audio_track(self):
        path = os.path.join(self.tmp_dir, "silent.mp4")
        _make_test_video(path, seconds=2, with_audio=False)
        with self.assertRaises(RuntimeError):
            extract_module.probe_media_info(path)

    def test_raises_for_nonexistent_file(self):
        with self.assertRaises(RuntimeError):
            extract_module.probe_media_info(os.path.join(self.tmp_dir, "does_not_exist.mp4"))


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg/ffprobe not found on PATH")
class TestExtractAudio(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="extract_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_single_pass_path_for_short_video(self):
        video_path = os.path.join(self.tmp_dir, "short.mp4")
        audio_path = os.path.join(self.tmp_dir, "audio.wav")
        _make_test_video(video_path, seconds=4, with_audio=True)
        extract_module.extract_audio(video_path, audio_path, duration=4, chunk_seconds=600, max_workers=2)
        self.assertTrue(os.path.exists(audio_path))
        self.assertAlmostEqual(_wav_duration(audio_path), 4, delta=0.5)

    def test_chunked_path_produces_correct_total_duration(self):
        # Use a real (if short) video and temporarily lower the chunking
        # threshold, so this exercises the exact same ThreadPoolExecutor +
        # concat code path a real multi-hour video would take, without
        # this test itself needing to process one.
        video_path = os.path.join(self.tmp_dir, "long.mp4")
        audio_path = os.path.join(self.tmp_dir, "audio.wav")
        video_seconds = 6
        _make_test_video(video_path, seconds=video_seconds, with_audio=True)

        original_threshold = extract_module._MIN_DURATION_FOR_CHUNKED_EXTRACT_SECONDS
        extract_module._MIN_DURATION_FOR_CHUNKED_EXTRACT_SECONDS = 1
        try:
            extract_module.extract_audio(
                video_path, audio_path, duration=video_seconds, chunk_seconds=2, max_workers=3
            )
        finally:
            extract_module._MIN_DURATION_FOR_CHUNKED_EXTRACT_SECONDS = original_threshold

        self.assertTrue(os.path.exists(audio_path))
        # 6s split into 2s ranges -> 3 parallel parts, concatenated back
        # into one continuous file spanning the original duration.
        self.assertAlmostEqual(_wav_duration(audio_path), video_seconds, delta=0.5)

    def test_falls_back_to_single_pass_when_duration_unknown(self):
        video_path = os.path.join(self.tmp_dir, "short2.mp4")
        audio_path = os.path.join(self.tmp_dir, "audio2.wav")
        _make_test_video(video_path, seconds=3, with_audio=True)
        extract_module.extract_audio(video_path, audio_path, duration=None)  # None -> always single pass
        self.assertTrue(os.path.exists(audio_path))


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg/ffprobe not found on PATH")
class TestCutChunkAndSilenceDetect(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cutchunk_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_tone(self, seconds: float) -> str:
        path = os.path.join(self.tmp_dir, "audio.wav")
        cmd = [
            FFMPEG_BIN, "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ar", "16000", "-ac", "1", path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return path

    def test_cut_chunk_produces_the_file(self):
        audio_path = self._make_tone(10)
        out_path = os.path.join(self.tmp_dir, "chunk.wav")
        transform_module.cut_chunk(audio_path, start=2, end=6, out_path=out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_detect_silences_finds_inserted_silence_gap(self):
        # 1s tone, 2s near-silence, 1s tone - detect_silences should find
        # the gap in the middle.
        path = os.path.join(self.tmp_dir, "gapped.wav")
        filt = (
            "sine=frequency=440:duration=1[a1];"
            "anullsrc=r=16000:cl=mono:d=2[a2];"
            "sine=frequency=440:duration=1[a3];"
            "[a1][a2][a3]concat=n=3:v=0:a=1[aout]"
        )
        cmd = [FFMPEG_BIN, "-y", "-filter_complex", filt, "-map", "[aout]", "-ar", "16000", path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)

        silences = transform_module.detect_silences(path, noise_db=-30, min_dur=0.5)
        self.assertTrue(len(silences) >= 1, "expected at least one detected silence gap")
        start, end = silences[0]
        self.assertAlmostEqual(start, 1.0, delta=0.3)
        self.assertAlmostEqual(end, 3.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
