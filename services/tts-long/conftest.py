"""Empty on purpose: its existence is what puts the repo root on sys.path.

pytest prepends the directory holding the rootmost conftest, which is how
`tests/test_conformance.py` can `import app.main`. Without this file pytest
would prepend `tests/` instead and the shared suite would collect nothing it
could build an app from.
"""

import pytest


def pytest_configure(config):
    """Register the marker that names a test needing the real weights.

    WHY A MARKER AND NOT A skipif. `skipif` on "is there a GPU" would make the
    test silently absent everywhere it matters and silently absent everywhere
    it does not, which is how a check ends up never having run anywhere --
    exactly the shape of `chatterbox-cpu`, a rung nothing ever offered a job.
    Deselected by default and reachable by name is a different thing: `pytest
    -m checkpoint` is in the README, the count of deselected tests is printed
    on every run, and the reason is in the test's own docstring.
    """
    config.addinivalue_line(
        "markers",
        "checkpoint: needs the real model weights on a real card. Deselected "
        "by default because no test in this tree loads a checkpoint -- the "
        "suite monkeypatches Synth._speak, which is the one method that "
        "touches a model. Run with: pytest -m checkpoint")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        # An explicit -m is the caller saying what they want, including
        # `-m checkpoint`. Second-guessing it here is how a selection becomes
        # unreachable.
        return
    deselected = [i for i in items if i.get_closest_marker("checkpoint")]
    if not deselected:
        return
    for item in deselected:
        items.remove(item)
    config.hook.pytest_deselected(items=deselected)
