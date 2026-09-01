"""Empty on purpose: its existence is what puts the repo root on sys.path.

pytest prepends the directory holding the rootmost conftest, which is how
`tests/test_conformance.py` can `import app.main`. Without this file pytest
would prepend `tests/` instead and the shared suite would collect nothing it
could build an app from.
"""
