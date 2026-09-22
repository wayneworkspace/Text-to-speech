# process/tests/ - test suite

Grouped the way the pipeline itself is grouped, so it is easy to tell what
needs what to run:

- **`unit/`** - fast, pure logic, no external processes and no heavy
  dependencies. Covers: `.env` parsing + the fail-fast chunk-settings
  validator (`test_config.py`), the batch lock + crash-recovery state
  machine (`test_batch_state.py`), speaker-matching math and name
  resolution (`test_speaker_id.py`), chunk-boundary planning + timestamp
  formatting (`test_transform_chunking.py`), the final Markdown writer
  (`test_load.py`), speaker-overlap assignment (`test_diarize.py`), the
  LLM transcript-correction step with the `anthropic` client mocked
  (`test_correct_transcript.py`), `batch.py`'s pure helpers - video
  discovery, stable-file detection, orphaned temp-dir cleanup, the
  per-video/per-run logger (`test_batch_helpers.py`) - and `main.py`'s
  `run_pipeline()` orchestration with every stage mocked: call order, the
  two API-key gates, temp-dir cleanup (including when a stage raises), and
  the return value (`test_run_pipeline.py`). Safe to run anywhere,
  including a sandboxed review environment with no GPU.
- **`integration/`** - calls real `ffmpeg`/`ffprobe` against small
  synthetic clips generated on the fly with ffmpeg's `lavfi` virtual
  inputs (no fixture files committed to the repo, nothing to download).
  Covers `probe_media_info`, both the single-pass and the new parallel
  chunked path of `extract_audio`, `cut_chunk`, and `detect_silences`.
  Skips itself cleanly (not a failure) if ffmpeg is not on PATH.

## What is NOT covered here (needs your real machine)

Whisper transcription, pyannote diarization, the speaker-embedding model,
and real calls to the Claude API all need `torch`/`whisper`/`pyannote.audio`,
a HuggingFace token, an Anthropic API key, and ideally a GPU - none of
which are available in the sandbox this suite was written and run in. Every
function that makes one of those real calls is instead exercised with it
mocked (`test_correct_transcript.py` fakes the `anthropic` module itself
via `sys.modules`; `test_run_pipeline.py` fakes every pipeline stage
function) - this covers the *logic around* those calls (prompts built
correctly, responses parsed and bounds-checked, errors always falling back
instead of raising, stage order, API-key gating, temp-dir cleanup) but not
whether the real model/API actually behaves as expected on real audio.
`main.py`'s `run_pipeline()` itself is also fully unit-tested despite
needing `tkinter` to run as a GUI - see "main.py imports tkinter lazily"
below. Those code paths were reviewed by hand instead; see CODE_REVIEW.md,
section 9, for exactly what is and is not automatically verified.

**`main.py` imports `tkinter` lazily** (inside `main()`, not at module
top-level - and so does its one `from pipeline.gui import App` import),
specifically so this file stays importable, and `run_pipeline()`
unit-testable, in an environment without `tkinter` installed (like the
sandbox this suite runs in). Only actually launching the GUI needs it;
this has no effect on normal `python main.py` usage on a machine with a
standard Python install.

## Running

This project targets Python 3.13+, where a plain directory is already an
importable package (PEP 420 namespace packages) - so nothing under
`tests/` has an `__init__.py`. One consequence: `unittest`'s CLI
`discover` happily treats a namespace-package directory as its *start*
directory, but does not recurse into a namespace-package *subdirectory* to
find nested tests - so a single `discover -s tests` silently finds "0
tests" now that both `tests/unit/` and `tests/integration/` lack
`__init__.py` too. `run_all.py` below exists specifically to work around
that (it calls `discover()` on each group directory directly, which does
work, and merges the results) - use it instead of `discover -s tests`.

From the `process/` folder, everything in one command:

    python tests/run_all.py -v

One group at a time (this form doesn't need `run_all.py` - discovering
directly inside a namespace-package leaf directory works fine):

    python -m unittest discover -s tests/unit -p "test_*.py" -v
    python -m unittest discover -s tests/integration -p "test_*.py" -v

A single file:

    python -m unittest tests.unit.test_batch_state -v

No extra dependencies to install - everything here uses the standard
library's `unittest`, plus `numpy` (already a project dependency) for the
speaker-matching math tests.
