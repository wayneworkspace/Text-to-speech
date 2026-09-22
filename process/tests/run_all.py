"""
Convenience entry point: run every test under both process/tests/unit/ and
process/tests/integration/ in a single command.

Why this exists: process/tests/ intentionally has no __init__.py files
(this project targets Python 3.13+, where a directory does not need one to
be importable - see PEP 420 namespace packages). Direct imports work fine
either way, and `python -m unittest discover -s tests/unit` (or
.../integration) works fine on its own - but unittest's CLI `discover`
does not recurse into a namespace-package *subdirectory* to find nested
tests, so the single combined `discover -s tests` silently finds "0 tests"
once tests/unit/ and tests/integration/ both lack __init__.py. This script
works around that by discovering each group directly (which does work)
and merging the results into one suite/run.

Run (from the process/ folder):
    python tests/run_all.py         # quiet
    python tests/run_all.py -v      # verbose, one line per test
"""
import os
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROCESS_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
sys.path.insert(0, _PROCESS_DIR)

_GROUPS = ("unit", "integration")


def load_all_tests() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for group in _GROUPS:
        group_dir = os.path.join(_TESTS_DIR, group)
        # top_level_dir=group_dir (not _TESTS_DIR) is what makes this work
        # without an __init__.py: it tells discover() to treat group_dir
        # itself as the (namespace-package) root, matching the same
        # command that works fine standalone: `discover -s tests/<group>`.
        suite.addTests(loader.discover(group_dir, pattern="test_*.py", top_level_dir=group_dir))
    return suite


if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(load_all_tests())
    sys.exit(0 if result.wasSuccessful() else 1)
