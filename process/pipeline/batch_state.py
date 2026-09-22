"""
BATCH_STATE: shared state for process/batch.py, the multi-video batch
runner. Two independent pieces live here:

1. A simple PID-based lock (acquire_lock/release_lock) so two batch.py
   processes never run at the same time and fight over the same GPU.
2. A per-video status table (load_state/save_state/mark_*) so a batch run
   that gets interrupted (crash, power loss, killed process) can resume
   later without reprocessing videos that already finished, and without
   permanently losing track of videos that were interrupted mid-way.

See CODE_REVIEW.md (sections 7-9) for the design this implements and why
each piece exists. In particular: a naive version of this - no staleness
check on the lock, no recovery of "running" entries - can deadlock itself
forever after a single crash. Both failure modes are handled explicitly
below, not as an afterthought.
"""
import json
import os
import time

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# A single video is generously assumed to never legitimately take longer
# than this. Used only as a second line of defense for the lock check below
# (the PID check is the primary one) - see acquire_lock().
_MAX_LOCK_AGE_SECONDS = 6 * 3600

# After this many failed attempts at the same video, stop retrying it
# automatically - a permanently broken file (corrupt, unsupported codec...)
# would otherwise be retried forever, once per cron cycle - and leave it
# for a human to check instead.
MAX_RETRIES = 3


# --------------------------------------------------------------------------
# Lock: only one batch.py process at a time (they would otherwise fight
# over the same GPU).
# --------------------------------------------------------------------------

def acquire_lock(lock_path: str, log) -> bool:
    """
    Try to become the only running batch.py. Returns True if the lock was
    acquired (safe to proceed), False if another instance is genuinely
    still running right now (caller should just exit quietly).

    A lock file left behind by a crashed process is detected two ways: its
    PID is no longer alive, or it is simply older than any single video
    could plausibly take. Either way it is treated as stale and cleared
    automatically, with a log message so this is never silent - the exact
    failure mode this project hit once already with a stale `.git/index.lock`.
    """
    if os.path.exists(lock_path):
        pid_alive, age_seconds = True, 0.0
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            pid_alive = _pid_is_alive(info.get("pid"))
            age_seconds = time.time() - info.get("started_at_epoch", 0)
        except (json.JSONDecodeError, OSError):
            # A lock file that cannot even be parsed cannot possibly be a
            # real, live lock - treat it as garbage and move on.
            pid_alive, age_seconds = False, _MAX_LOCK_AGE_SECONDS + 1

        if pid_alive and age_seconds < _MAX_LOCK_AGE_SECONDS:
            return False  # someone else is genuinely running right now - back off

        log(
            f"Found a leftover lock file (pid_alive={pid_alive}, "
            f"age={age_seconds:.0f}s) - a previous run must have crashed "
            "without cleaning up. Clearing it and continuing."
        )
        os.remove(lock_path)

    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started_at_epoch": time.time()}, f)
    return True


def release_lock(lock_path: str) -> None:
    """Remove our own lock file. Safe to call even if it is already gone."""
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _pid_is_alive(pid) -> bool:
    """Best-effort check for whether a process with this PID is still running."""
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # psutil is required (see requirements.txt) but if it is somehow
        # missing, fail safe: assume the PID might still be alive and lean
        # on the age check in acquire_lock() instead of risking two
        # batch.py instances running at once.
        return True


# --------------------------------------------------------------------------
# Per-video state: what has been processed, what failed, what is left.
# --------------------------------------------------------------------------

def load_state(state_path: str) -> dict:
    """Load the per-video status table. Returns {} if it does not exist yet."""
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def recover_interrupted(state: dict, log) -> dict:
    """
    Call this once, right after acquire_lock() succeeds. Because only one
    batch.py can hold the lock at a time, if we just acquired it, nothing
    else can legitimately be mid-processing right now - so any video still
    marked "running" here is a leftover from a run that crashed before it
    could mark that video "done" or "failed". Reset it to "pending" (and
    count it as a used attempt) so it gets tried again instead of being
    stuck in "running" forever.
    """
    stuck = [name for name, info in state.items() if info.get("status") == STATUS_RUNNING]
    for name in stuck:
        state[name]["status"] = STATUS_PENDING
        state[name]["retry_count"] = state[name].get("retry_count", 0) + 1
    if stuck:
        log(f"Recovering {len(stuck)} interrupted video(s) from a previous run: {stuck}")
    return state


def mark_running(state: dict, video_name: str) -> None:
    state[video_name] = state.get(video_name, {})
    state[video_name]["status"] = STATUS_RUNNING
    state[video_name]["started_at_epoch"] = time.time()


def mark_done(state: dict, video_name: str, output_path: str) -> None:
    state[video_name]["status"] = STATUS_DONE
    state[video_name]["output_path"] = output_path
    state[video_name]["finished_at_epoch"] = time.time()
    state[video_name].pop("error", None)


def mark_failed(state: dict, video_name: str, error: str) -> None:
    """
    Record a failed attempt and bump its retry count. Left as "pending" so
    the next batch run retries it automatically - should_process() is what
    turns this permanently into "failed" once MAX_RETRIES is exceeded, so
    there is a single place that decides "give up on this one".
    """
    retry_count = state[video_name].get("retry_count", 0) + 1
    state[video_name]["retry_count"] = retry_count
    state[video_name]["status"] = STATUS_PENDING
    state[video_name]["error"] = error
    state[video_name]["failed_at_epoch"] = time.time()


def should_process(state: dict, video_name: str, log) -> bool:
    """
    True if this video still needs (another) attempt. False for "done"
    (already finished) or a video that has exceeded MAX_RETRIES - the
    latter is decided right here and promoted to "failed" permanently, so
    a single broken file cannot be retried forever, once per cron cycle.
    """
    info = state.get(video_name)
    if info is None:
        return True
    if info.get("status") == STATUS_DONE:
        return False
    if info.get("status") == STATUS_FAILED:
        return False
    if info.get("retry_count", 0) > MAX_RETRIES:
        info["status"] = STATUS_FAILED
        log(f"Giving up on '{video_name}' after {MAX_RETRIES} failed attempts - needs a manual look.")
        return False
    return True
