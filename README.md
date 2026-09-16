# Video-to-Text Transcription Pipeline

A local pipeline that turns the spoken content of `.mp4` videos into timestamped Markdown transcripts, using FFmpeg for audio processing and a local Whisper model for speech recognition.

**Techstack:**

Python | FFmpeg | OpenAI Whisper | Tkinter | python-dotenv

## Overview

**Problem** — Manually transcribing spoken content from recordings (lectures, meetings, interviews) is slow and tedious.

**Solution** — A small desktop tool: pick an `.mp4` file through a simple GUI, and the pipeline extracts the audio, splits it into speech-friendly chunks, transcribes it locally with Whisper, and writes a clean, timestamped Markdown transcript — fully offline after the first run.

**Data Flow** — MP4 video (GUI file picker) → raw audio (FFmpeg) → silence-aware audio chunks → text segments (Whisper) → timestamped Markdown file.

## Architecture

Source → Ingestion → Processing → Storage

- **Source** — the user selects an `.mp4` video through a Tkinter file dialog (`pipeline/gui.py`).
- **Ingestion** — FFmpeg/ffprobe read the video's metadata and extract a mono 16kHz WAV audio track (`pipeline/extract.py`).
- **Processing** — the audio is split into chunks at silence boundaries and transcribed chunk-by-chunk with Whisper; timestamps are re-aligned to match the original video (`pipeline/transform.py`).
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
        ├── load.py
        ├── utils.py
        └── gui.py
```

`main.py` is the orchestrator: it calls `extract.py` → `transform.py` → `load.py` in order and wires the GUI to that pipeline. Each pipeline module only knows about its own stage.

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

## Limitations

- Since audio is split into chunks, a word or two can occasionally be lost right at a chunk boundary if no good nearby silence was found. The chunk planner always prefers cutting at the closest detected silence to minimize this.
- Transcription accuracy depends on source audio quality (background noise, overlapping speakers, accents, domain-specific terms...).
- Processing can take longer than the video's actual runtime on CPU-only machines.
- Running the pipeline twice on the same day for the same video **overwrites** the previous transcript (same filename `[dd-mm-yy] - <video_name> - transcript.md`).
