# Video-to-Text Transcription Pipeline

A local pipeline that turns the spoken content of `.mp4` videos into timestamped Markdown transcripts, using FFmpeg for audio processing and a local Whisper model for speech recognition.

<<<<<<< HEAD
Techstack
Python | FFmpeg | OpenAI Whisper | pyannote.audio | Anthropic API | Tkinter | python-dotenv
=======
**Techstack:**

Python | FFmpeg | OpenAI Whisper | Tkinter | python-dotenv
>>>>>>> ebbaf3b4b2387dd210cea749a22371680b46cc5c

## Overview

**Problem** — Manually transcribing spoken content from recordings (lectures, meetings, interviews) is slow and tedious, and a plain flat transcript is still hard to use as an input to tools like NotebookLM (no idea who said what, no topic structure).

**Solution** — A small desktop tool: pick an `.mp4` file through a simple GUI, and the pipeline extracts the audio, splits it into speech-friendly chunks, transcribes it locally with Whisper, optionally figures out who said what (speaker diarization) and where the discussion changes topic (via a single Claude API call), then writes a clean, timestamped Markdown transcript — fully offline after the first run except for the two optional enrichment steps.

**Data Flow** — MP4 video (GUI file picker) → raw audio (FFmpeg) → silence-aware audio chunks → text segments (Whisper) → speaker-labeled segments (pyannote.audio, optional) → topic markers (Claude API, optional) → timestamped Markdown file.

## Architecture

Source → Ingestion → Processing → Storage

- **Source** — the user selects an `.mp4` video through a Tkinter file dialog (`pipeline/gui.py`).
- **Ingestion** — FFmpeg/ffprobe read the video's metadata and extract a mono 16kHz WAV audio track (`pipeline/extract.py`).
- **Processing** — the audio is split into chunks at silence boundaries and transcribed chunk-by-chunk with Whisper, using an optional domain-vocabulary hint and exposing a confidence flag per segment; timestamps are re-aligned to match the original video (`pipeline/transform.py`). Two steps here are optional and run only if their credentials are set: speaker diarization (`pipeline/diarize.py`) and topic segmentation via a single Claude API call (`pipeline/enrich.py`).
- **Storage** — the final transcript is written as a Markdown file under `output/`, one file per run (`pipeline/load.py`).

There is no "Serving" layer: this is a local tool, not a service — the output is a static file the user opens directly.

## Project Structure

```text
project/
├── README.md
├── requirements.txt
├── .env
├── output/
└── process/
    ├── main.py
    ├── config.py
    └── pipeline/
        ├── extract.py
        ├── transform.py
        ├── diarize.py
        ├── enrich.py
        ├── load.py
        ├── utils.py
        └── gui.py
```

`main.py` is the orchestrator: it calls `extract.py` → `transform.py` → (`diarize.py` + `enrich.py`, if configured) → `load.py` in order and wires the GUI to that pipeline. Each pipeline module only knows about its own stage.

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

All tunable settings live in a `.env` file at the project root, loaded by `process/config.py`:

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
| `DOMAIN_VOCABULARY` | Optional. Comma-separated words/phrases (or a sample sentence) hinting Whisper toward domain-specific terms, names, or jargon it would otherwise mis-transcribe. Leave empty to skip. |
| `CONFIDENCE_THRESHOLD` | Whisper segments scoring below this (its own `avg_logprob`) are flagged with ⚠️ in the output, as lines worth double-checking. |
| `HUGGINGFACE_TOKEN` | Optional. Enables speaker diarization (who said what) via pyannote.audio. Leave empty to skip diarization entirely — see "Getting a HuggingFace token" below. |
| `ANTHROPIC_API_KEY` | Optional. Enables automatic topic segmentation via the Claude API. Leave empty to skip — see "Getting an Anthropic API key" below. |
| `ANTHROPIC_MODEL` | Which Claude model to use for topic segmentation (default: `claude-sonnet-5`). |

## Getting a HuggingFace token (optional — for speaker diarization)

Speaker diarization uses a gated model, so it needs a free HuggingFace account and a token:

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Open [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept the user conditions on that page (it's free, CC-BY-4.0 license — it just asks for contact info).
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new access token (the default "Read" permission is enough).
4. Paste it into `.env` as `HUGGINGFACE_TOKEN=hf_...`.

If this is left empty, `main.py` logs a notice and skips diarization — transcription still runs normally, just without speaker labels.

## Getting an Anthropic API key (optional — for topic segmentation)

Topic segmentation makes one Claude API call per video. This is a separate, pay-as-you-go API account — not the same as a claude.ai subscription:

1. Go to [platform.claude.com](https://platform.claude.com) and sign up or log in.
2. Add billing under your account (API usage is billed per token; a single topic-segmentation call is inexpensive).
3. Create a key at [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys).
4. Paste it into `.env` as `ANTHROPIC_API_KEY=sk-ant-...`.

If this is left empty, `main.py` logs a notice and skips topic segmentation — the transcript is written as one flat list with no topic headings.

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
- **`ModuleNotFoundError: No module named 'tkinter'`** — install the tkinter package for your Python (Linux: `sudo apt install python3-tk`).
- **`ModuleNotFoundError: No module named 'pipeline'` or `'config'`** — you're not running `main.py` correctly. Run it from the project root (`python process/main.py`) or `cd process && python main.py`; don't copy `main.py` out on its own without `pipeline/`, `config.py`, and `.env`.
- **Error mentioning the `whisper` package** (e.g. missing `load_model`) — you may have the wrong PyPI package installed; `whisper` and `openai-whisper` both use the `whisper` import name. Fix:
  ```
  pip uninstall whisper
  pip install openai-whisper
  ```
- **Running very slowly / hanging** — on CPU, Whisper is much slower than real-time, especially with a large model. Try `MODEL_SIZE=small` or `base` in `.env` for more speed (at some accuracy cost).
- **Non-.mp4 input** — FFmpeg reads most common video/audio formats; the file dialog also has an "All files" option if needed.
- **401/403 error while loading the diarization model** — your `HUGGINGFACE_TOKEN` is missing/invalid, or you haven't accepted the terms on the [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) page yet. See "Getting a HuggingFace token" above.
- **Topic segmentation always skipped / "Claude call failed" in the log** — check `ANTHROPIC_API_KEY` in `.env` is set and valid, and that the account has billing enabled. This never stops the transcript from being written — it just comes out without topic headings.

## Limitations

- Since audio is split into chunks, a word or two can occasionally be lost right at a chunk boundary if no good nearby silence was found. The chunk planner always prefers cutting at the closest detected silence to minimize this.
- Transcription accuracy depends on source audio quality (background noise, overlapping speakers, accents, domain-specific terms...). `DOMAIN_VOCABULARY` and the ⚠️ confidence flag help, but don't eliminate this.
- Processing can take longer than the video's actual runtime on CPU-only machines.
- Running the pipeline twice on the same day for the same video **overwrites** the previous transcript (same filename `[dd-mm-yy] - <video_name> - transcript.md`).
- Speaker labels (`SPEAKER_00`, `SPEAKER_01`, ...) are anonymous — pyannote only tells voices apart, it has no idea of real names. Two speakers with very similar-sounding voices can occasionally get merged into one label or split inconsistently.
- Topic segmentation is generated by an LLM reading the transcript once — it's a reasonable approximation of topic boundaries, not a guaranteed-correct one, and each run costs a small amount of Claude API usage.
- Both speaker diarization and topic segmentation are skipped gracefully (with a log message, not an error) if their respective credentials aren't set in `.env`.
