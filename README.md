# Video-to-Text Transcription Pipeline

A local pipeline that turns the spoken content of `.mp4` videos into timestamped Markdown transcripts, using FFmpeg for audio processing and a local Whisper model for speech recognition.

**Techstack:**

Python | FFmpeg | OpenAI Whisper | pyannote.audio | Anthropic API | Tkinter | python-dotenv

## Overview

**Problem** — Manually transcribing spoken content from recordings (lectures, meetings, interviews) is slow and tedious, and a plain flat transcript is still hard to use as an input to tools like NotebookLM (no idea who said what, no topic structure).

**Solution** — A small desktop tool: pick an `.mp4` file through a simple GUI, and the pipeline extracts the audio, splits it into speech-friendly chunks, transcribes it locally with Whisper, optionally corrects likely mis-transcribed words using context (proper nouns, foreign technical terms spoken mid-sentence, domain jargon), figures out who said what (speaker diarization), and works out where the discussion changes topic (two more Claude API calls, both optional), then writes a clean, timestamped Markdown transcript — fully offline after the first run except for the optional enrichment steps. For processing many videos unattended (e.g. a folder that gets new recordings dropped into it every so often), `process/batch.py` runs the same pipeline as a resumable, crash-safe CLI job instead of one file through the GUI — see "Batch processing many videos" below.

**Data Flow** — MP4 video (GUI file picker, or a folder scanned by `batch.py`) → raw audio (FFmpeg) → silence-aware audio chunks → text segments (Whisper) → corrected segments (Claude API, optional) → speaker-labeled segments (pyannote.audio, optional) → topic markers (Claude API, optional) → timestamped Markdown file.

## Architecture

Source → Ingestion → Processing → Storage

- **Source** — the user selects an `.mp4` video through a Tkinter file dialog (`pipeline/gui.py`), or `process/batch.py` scans `INPUT_DIR` recursively for video files.
- **Ingestion** — FFmpeg/ffprobe read the video's metadata and extract a mono 16kHz WAV audio track, splitting long videos into parallel chunks for speed (`pipeline/extract/`).
- **Processing** — the audio is split into chunks at silence boundaries and transcribed chunk-by-chunk with Whisper, using an optional domain-vocabulary hint and exposing a confidence flag per segment; timestamps are re-aligned to match the original video (`pipeline/transform/`). Four steps here are optional and run only if their credentials are set: LLM-based transcript correction, speaker diarization, speaker recognition, and topic segmentation - the last two via the Claude API (all under `pipeline/enrich/`).
- **Storage** — the final transcript is written as a Markdown file under `output/`, one file per run (`pipeline/load/`).

There is no "Serving" layer: this is a local tool, not a service — the output is a static file the user opens directly.

Each ETL stage above is its own subfolder under `pipeline/` (mirroring this list) — no `__init__.py` files (this project targets Python 3.13+, where a plain directory is already an importable namespace package, PEP 420). `main.py` imports each stage's public functions straight from its module, e.g. `from pipeline.extract.extract import probe_media_info, extract_audio`, and finding "the diarization code" means opening `pipeline/enrich/diarize.py`, not scanning a flat list of files.

## Project Structure

```text
project/
├── README.md
├── CODE_REVIEW.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── data/
│   └── speaker_profiles.json   # enrolled speaker voice database (not tracked by git)
├── output/                     # written transcripts (main.py and batch.py both write here)
├── state/                      # batch.py's progress table + lock file (not tracked by git)
├── logs/                       # batch.py's per-run and per-video logs (not tracked by git)
└── process/
    ├── main.py                 # GUI entry point - one video at a time
    ├── batch.py                # CLI entry point - many videos unattended (e.g. Task Scheduler)
    ├── config.py
    ├── enroll_speaker.py
    ├── pipeline/
    │   ├── extract/             # EXTRACT stage: FFmpeg metadata + audio extraction
    │   │   └── extract.py
    │   ├── transform/           # TRANSFORM stage: silence-aware chunking + Whisper
    │   │   └── transform.py
    │   ├── enrich/               # ENRICH stage (optional): correction, diarization, speaker recognition, topics
    │   │   ├── correct.py
    │   │   ├── diarize.py
    │   │   ├── speaker_id.py
    │   │   └── enrich.py
    │   ├── load/                 # LOAD stage: write the Markdown transcript
    │   │   └── load.py
    │   ├── batch_state.py       # lock file + per-video state machine used by batch.py
    │   ├── utils.py             # shared helpers (FFmpeg paths, timestamp formatting)
    │   └── gui.py
    └── tests/                    # unit + integration test suite (see "Running the tests")
        ├── unit/
        └── integration/
```

`main.py` is the orchestrator: it calls `extract/` → `transform/` → (`enrich/`, if configured) → `load/` in order and wires the GUI to that pipeline. `batch.py` calls the same `run_pipeline()` in `main.py` once per video found under `INPUT_DIR`, adding a lock file, a per-video progress table, and crash recovery on top (`pipeline/batch_state.py`) — see below.

## Quick Start

### Prerequisites

- Python 3.9+ (3.10–3.12 recommended).
- **FFmpeg** (`ffmpeg` + `ffprobe`) installed separately and available on `PATH` — this is not a pip package. See "Installing FFmpeg" below.
- A GPU (NVIDIA + CUDA) makes transcription much faster, but CPU works too (slower).
- Internet access is required once, the first time a Whisper model is downloaded (`medium` ≈ 1.5GB). Every run after that is fully offline.

### Setup

```
pip install -r requirements.txt
python process/main.py
```

A small window opens — click **"Choose MP4 video..."**, pick a file, and the pipeline runs automatically with a live progress log. The transcript is written to `output/[dd-mm-yy] - <video_name> - transcript.md` (date = the day the pipeline ran).

You can also pass a file path directly to skip the dialog:
```
python process/main.py "path/to/video.mp4"
```

### Configuration (`.env`)

All tunable settings live in a `.env` file at the project root, loaded by `process/config.py`. Copy `.env.example` to `.env` and fill in what you need - `.env` itself is gitignored, so your keys never get committed:

| Key | Notes |
|---|---|
| `MODEL_SIZE` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large-v3`. Bigger = more accurate, but slower and heavier on RAM/VRAM. |
| `LANGUAGE` | Audio language code, e.g. `vi` for Vietnamese. Leave empty to let Whisper auto-detect the language. |
| `CHUNK_TARGET_SECONDS` | Target length of each audio chunk, in seconds. |
| `CHUNK_MIN_SECONDS` | Minimum chunk length before a silence-based cut is allowed. |
| `CHUNK_MAX_SECONDS` | Maximum chunk length; forces a hard cut if no good silence point is found nearby. |
| `SILENCE_NOISE_DB` | Loudness threshold (dBFS) below which audio counts as silence. |
| `SILENCE_MIN_DURATION` | Minimum duration (seconds) a quiet moment must last to count as silence. |
| `KEEP_TEMP_FILES` | `true` / `false` — keep the extracted `audio.wav` and `chunk_*.wav` files around for debugging. |
| `DOMAIN_VOCABULARY` | Optional. Comma-separated words/phrases (or a sample sentence) hinting Whisper toward domain-specific terms, names, or jargon it would otherwise mis-transcribe. Also passed as context to the transcript-correction step below, if `ANTHROPIC_API_KEY` is set. Leave empty to skip. |
| `CONFIDENCE_THRESHOLD` | Whisper segments scoring below this (its own `avg_logprob`) are flagged with ⚠️ in the output, as lines worth double-checking. |
| `HUGGINGFACE_TOKEN` | Optional. Enables speaker diarization (who said what) via pyannote.audio. Leave empty to skip diarization entirely — see "Getting a HuggingFace token" below. |
| `ANTHROPIC_API_KEY` | Optional. Enables two independent Claude API steps: transcript correction (fixes likely mis-transcribed words using context) and topic segmentation. Leave empty to skip both — see "Getting an Anthropic API key" below. |
| `ANTHROPIC_MODEL` | Which Claude model to use for transcript correction and topic segmentation (default: `claude-sonnet-5`). |
| `SPEAKER_PROFILES_PATH` | Optional. Where the enrolled speaker-voice database lives. Leave empty to use the default (`data/speaker_profiles.json`). |
| `SPEAKER_MATCH_THRESHOLD` | Cosine-similarity threshold (0-1) above which a voice is considered a match for an enrolled speaker (default `0.75`). Higher = stricter matching, more voices fall back to `Speaker 1`, `Speaker 2`... |
| `AUTO_UPDATE_SPEAKER_PROFILES` | `true` / `false` - if `true`, a confident match also refines that speaker's stored profile with the new sample. Leave `false` until you trust the matches you are getting (default `false`). |
| `EXTRACT_CHUNK_SECONDS` | How many seconds of video each parallel FFmpeg worker extracts at once for a long video (default `600`). Only kicks in past a length where parallelizing pays for itself — see "Faster audio extraction" below. |
| `EXTRACT_MAX_WORKERS` | Max number of parallel FFmpeg extraction workers (default `4`). Keep at or below your CPU's core count. |
| `INPUT_DIR` | `process/batch.py` only — folder to scan recursively for videos. Leave empty and `batch.py` does nothing; `main.py`'s GUI is unaffected either way. |
| `STATE_PATH` / `LOCK_PATH` / `LOG_DIR` | `process/batch.py` only — override where its progress table, lock file, and logs live. Leave empty for the defaults (`state/batch_state.json`, `state/.batch.lock`, `logs/`). |

## Getting a HuggingFace token (optional — for speaker diarization)

Speaker diarization uses a gated model, so it needs a free HuggingFace account and a token:

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Open [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept the user conditions on that page (it's free, CC-BY-4.0 license — it just asks for contact info).
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new access token (the default "Read" permission is enough).
4. Paste it into `.env` as `HUGGINGFACE_TOKEN=hf_...`.

If this is left empty, `main.py` logs a notice and skips diarization — transcription still runs normally, just without speaker labels.

## Speaker recognition across videos (optional, needs diarization)

Diarization alone only tells voices apart *within one video* — its `SPEAKER_00`, `SPEAKER_01` labels are anonymous cluster IDs that reset on every run. `pipeline/enrich/speaker_id.py` adds a second, optional layer on top: it computes a voiceprint (embedding) for each diarized voice and compares it against a small local database of people you've enrolled by name (`data/speaker_profiles.json`, not tracked by git). A close match is shown with that person's real name in the transcript; anything that doesn't match falls back to a stable `Speaker 1`, `Speaker 2`... label instead of the raw `SPEAKER_00`.

This uses the same `HUGGINGFACE_TOKEN` as diarization — no extra account setup needed.

**Enrolling someone:**

```
python process/enroll_speaker.py sample.wav "Nguyen Van A"
```

Use a short, clean recording of just that one person (20–60s, minimal background noise/overlap) for the most accurate result. You can also enroll from an already-diarized video by picking out one raw speaker label:

```
python process/enroll_speaker.py meeting.mp4 "Nguyen Van A" --speaker SPEAKER_00
```

Enrolling the same person again (from a different recording) refines their profile instead of replacing it — the new sample is averaged into the existing one.

Matching accuracy depends on recording quality and how distinct people's voices are — tune `SPEAKER_MATCH_THRESHOLD` in `.env` if you're seeing wrong matches (raise it) or known speakers not being recognized (lower it).

## Getting an Anthropic API key (optional — for transcript correction and topic segmentation)

The same `ANTHROPIC_API_KEY` unlocks two independent, optional steps, each making one Claude API call per video. This is a separate, pay-as-you-go API account — not the same as a claude.ai subscription:

1. Go to [platform.claude.com](https://platform.claude.com) and sign up or log in.
2. Add billing under your account (API usage is billed per token; each call here is inexpensive).
3. Create a key at [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys).
4. Paste it into `.env` as `ANTHROPIC_API_KEY=sk-ant-...`.

**Transcript correction** (`pipeline/enrich/correct.py`) runs right after Whisper, before anything else reads the text. It sends Claude the full transcript once — plus the video's filename and `DOMAIN_VOCABULARY`, if set, as context — and asks it to flag only the lines it's fairly confident were mis-transcribed (typically a proper noun, a foreign/technical term spoken mid-sentence, or domain jargon Whisper doesn't know) and return corrected text for just those. It is explicitly told not to rephrase, summarize, translate, or otherwise change meaning, and it returns an empty list on the common case where nothing looks wrong - so cost stays low and roughly constant regardless of transcript length, rather than growing with segment count. As with every other optional step here, a bad key, no network, or a malformed response never breaks the pipeline: it logs a notice and the transcript goes through unchanged.

**Topic segmentation** (`pipeline/enrich/enrich.py`) reads the (corrected) transcript once and returns a list of topic headings inserted into the Markdown output.

If `ANTHROPIC_API_KEY` is left empty, `main.py` logs a notice and skips both steps — the transcript is written as transcribed, with no corrections and no topic headings.

## Batch processing many videos (optional)

`process/main.py`'s GUI is built for one video at a time. For a folder that regularly receives new recordings — e.g. 80 files dropped in over time, not all at once — `process/batch.py` is a CLI entry point that scans a folder and processes whatever is new, designed to be triggered on a schedule (Windows Task Scheduler, cron, ...) rather than run by hand:

```
python process/batch.py
```

Set `INPUT_DIR` in `.env` first — `batch.py` scans it recursively for `.mp4`/`.mov`/`.mkv`/`.avi`/`.webm` files and processes each one through the same `run_pipeline()` as `main.py`, writing to `output/` the same way. What it adds on top (all in `pipeline/batch_state.py`):

- **Never processes the same video twice** — a `state/batch_state.json` table tracks each video as `pending` / `running` / `done` / `failed`, so re-running `batch.py` (e.g. the next scheduled trigger) only picks up new or unfinished videos.
- **Safe to trigger on a schedule that might overlap itself** — a lock file (`state/.batch.lock`) stops two `batch.py` runs from processing videos at the same time and fighting over the GPU. If a previous run crashed and left the lock behind, the next run detects this (the lock's PID is no longer alive, or it is simply older than any single video could plausibly take) and clears it automatically, logging that it did so.
- **Recovers from being killed mid-video** — if `batch.py` is interrupted while a video is `running` (killed process, power loss, a Task Scheduler timeout), the *next* run resets that video back to `pending` instead of leaving it stuck forever, and retries it.
- **Gives up on a permanently broken file instead of retrying it forever** — after `MAX_RETRIES` (3) failed attempts at the same video, it's marked `failed` for good and skipped on future runs, so one corrupt file doesn't eat a retry every single cron cycle. Check `logs/` for why.
- **Skips a file that is still being copied in** — before processing, it checks the file's size is stable for a few seconds, so a video that's still mid-upload/mid-copy into `INPUT_DIR` isn't picked up half-written.
- **Cleans up after a crash** — sweeps any leftover temp folders from an interrupted run's extracted audio at the start of every batch.

Logs: one file per batch run plus one file per video, both under `logs/` (also gitignored, like `state/`), so a single failure among many videos is easy to find without reading a combined log.

**Scheduling it (Windows):** create a Basic Task in Task Scheduler that runs `python C:\path\to\process\batch.py` every 30–60 minutes (adjust `pipeline/batch_state.py`'s `MAX_RETRIES`/lock behavior only if you have a reason to — the defaults are meant to be safe as-is). Since videos arrive irregularly, frequent polling on a schedule is simpler and more robust here than an event-driven watcher for a single-machine, single-GPU setup — see `CODE_REVIEW.md`, section 7, for the reasoning.

## Faster audio extraction for long videos

For a video at or above `EXTRACT_CHUNK_SECONDS` (10 minutes by default), `pipeline/extract/extract.py` splits audio extraction into `EXTRACT_CHUNK_SECONDS`-long time ranges and runs FFmpeg on each range in parallel (up to `EXTRACT_MAX_WORKERS` at once) instead of one FFmpeg process reading the file start to finish, then stitches the parts back together. FFmpeg's audio decode is single-threaded per process, so this uses otherwise-idle CPU cores while the GPU is busy transcribing a *different* video — cutting wall-clock extraction time roughly by a factor of `EXTRACT_MAX_WORKERS` on long recordings. Shorter videos, or any video whose duration couldn't be determined, always use a single pass — splitting would cost more (starting several FFmpeg processes, then concatenating) than it could save. This is automatic; there is nothing to turn on beyond the two `.env` keys above, and both have sensible defaults.

## Running the tests

A grouped `unittest` suite lives under `process/tests/` (`unit/` for fast, dependency-free logic; `integration/` for tests against real FFmpeg using synthetic clips generated on the fly — nothing to install or download beyond FFmpeg itself). From the `process/` folder, run everything in one command:

```
python tests/run_all.py -v
```

Or discover one group directly (works because neither `tests/`, `tests/unit/`, nor `tests/integration/` has an `__init__.py` — see `process/tests/README.md` for why `run_all.py` exists instead of a single `discover -s tests`):

```
python -m unittest discover -s tests/unit -p "test_*.py" -v
python -m unittest discover -s tests/integration -p "test_*.py" -v
```

See `process/tests/README.md` for running a single file, and for exactly what is (and, since this sandbox has no GPU/Whisper/pyannote, is not) covered.

## Installing FFmpeg

**Windows**
- `winget install --id=Gyan.FFmpeg -e`, or Chocolatey: `choco install ffmpeg`.
- Or download a prebuilt release from https://www.gyan.dev/ffmpeg/builds/, unzip it, and add its `bin` folder to your `PATH`.
- Verify: open Command Prompt and run `ffmpeg -version`.

**macOS**
```
brew install ffmpeg
```

**Linux (Debian/Ubuntu)**
```
sudo apt update && sudo apt install ffmpeg
```

If `openai-whisper`'s default `torch` install is too heavy and your machine has **no NVIDIA GPU**, install the lighter CPU-only build first:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If `tkinter` is missing (some minimal Linux Python installs don't include it):
```
sudo apt install python3-tk
```

## Troubleshooting

- **"Could not find 'ffmpeg' on PATH"** — FFmpeg isn't installed or isn't on `PATH`. Revisit "Installing FFmpeg" above, and open a new terminal after installing.
- **`ModuleNotFoundError: No module named 'tkinter'`** — install the tkinter package for your Python (Linux: `sudo apt install python3-tk`). Only `main.py` (the GUI) needs this — `batch.py` does not.
- **`ModuleNotFoundError: No module named 'pipeline'` or `'config'`** — you're not running `main.py`/`batch.py` correctly. Run from the project root (`python process/main.py`) or `cd process && python main.py`; don't copy a script out on its own without `pipeline/`, `config.py`, and `.env`.
- **Error mentioning the `whisper` package** (e.g. missing `load_model`) — you may have the wrong PyPI package installed; `whisper` and `openai-whisper` both use the `whisper` import name. Fix:
  ```
  pip uninstall whisper
  pip install openai-whisper
  ```
- **Running very slowly / hanging** — on CPU, Whisper is much slower than real-time, especially with a large model. Try `MODEL_SIZE=small` or `base` in `.env` for more speed (at some accuracy cost).
- **Non-.mp4 input** — FFmpeg reads most common video/audio formats; the file dialog also has an "All files" option if needed.
- **401/403 error while loading the diarization model** — your `HUGGINGFACE_TOKEN` is missing/invalid, or you haven't accepted the terms on the [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) page yet. See "Getting a HuggingFace token" above.
- **Topic segmentation / transcript correction always skipped, or "Claude call failed" in the log** — check `ANTHROPIC_API_KEY` in `.env` is set and valid, and that the account has billing enabled. Both steps share this one key. This never stops the transcript from being written — it just comes out uncorrected and/or without topic headings.
- **`batch.py` says "Another batch.py run is already in progress - exiting."** — expected if a previous run is genuinely still processing a video; it will simply run again on the next schedule trigger. If no `batch.py` should actually be running, check `state/.batch.lock`'s age/PID — it should self-clear within `pipeline/batch_state.py`'s staleness window on the next run either way.

## Limitations

- Since audio is split into chunks, a word or two can occasionally be lost right at a chunk boundary if no good nearby silence was found. The chunk planner always prefers cutting at the closest detected silence to minimize this. The same class of boundary imprecision (tens of milliseconds, not full words) applies to the parallel audio-extraction chunking described in "Faster audio extraction" above, for the same reason (fast `-ss`-before-`-i` seeking).
- Transcription accuracy depends on source audio quality (background noise, overlapping speakers, accents, domain-specific terms...). `DOMAIN_VOCABULARY`, the ⚠️ confidence flag, and the optional transcript-correction step all help, but don't eliminate this - correction only catches errors the LLM can infer from context (a name, a term it recognizes as out of place), not systematic mis-hearings it has no way to detect from text alone.
- Processing can take longer than the video's actual runtime on CPU-only machines.
- Running the pipeline twice on the same day for the same video **overwrites** the previous transcript (same filename `[dd-mm-yy] - <video_name> - transcript.md`), whether run through `main.py` or `batch.py`.
- Speaker labels (`SPEAKER_00`, `SPEAKER_01`, ...) from diarization alone are anonymous — pyannote only tells voices apart, it has no idea of real names. Two speakers with very similar-sounding voices can occasionally get merged into one label or split inconsistently. Enrolling people via `pipeline/enrich/speaker_id.py` (see "Speaker recognition across videos" above) resolves known voices to real names, but matching is similarity-based, not exact — an unusually noisy recording or a wrongly-tuned `SPEAKER_MATCH_THRESHOLD` can still produce a wrong match or a missed one.
- Transcript correction and topic segmentation are both generated by an LLM reading the transcript once - reasonable approximations, not guaranteed-correct, and each run costs a small amount of Claude API usage.
- Transcript correction, speaker diarization, and topic segmentation are all skipped gracefully (with a log message, not an error) if their respective credentials aren't set in `.env`.
- `batch.py` gives up on a video after `MAX_RETRIES` (3) failed attempts and marks it permanently `failed` — a genuinely broken file needs a human to look at `logs/` and fix or remove it; it will not be retried automatically after that point.
