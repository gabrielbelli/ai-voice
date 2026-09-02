"""Present so that pytest puts the repository root on sys.path.

tests/test_conformance.py imports `app.main`, which is a package in this
directory rather than an installed distribution. Under pytest's default import
mode the rootdir is added to sys.path only when a conftest.py lives there, so
this file is the entire reason that import resolves. It has nothing else to do.
"""
