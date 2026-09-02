"""The entrypoint, exercised as a real /bin/sh script.

Three copies of this file exist today with three different holes, the same
pattern as auth.py, so the shared one is the union of the strictest rule from
each. Every test here names the hole it closes.

Only the non-root paths are exercised: the tests run as an ordinary user on a
developer's machine, and the root branches call setpriv and chown, which are
Linux-only and destructive. What CAN be tested here is every decision the
script makes before it drops privileges, which is where all three holes were.
The command is a stub on PATH that prints its own arguments, so the flags the
script appends are observable without running uvicorn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "voice-entrypoint.sh"


@pytest.fixture
def fake_uvicorn(tmp_path: Path) -> Path:
    """A `uvicorn` on PATH that reports the arguments it was given."""
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "uvicorn"
    stub.write_text('#!/bin/sh\necho "ARGS: $*"\n')
    stub.chmod(0o755)
    return binary


def run(*args: str, env: dict[str, str] | None = None,
        path: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/")}
    if path is not None:
        environment["PATH"] = f"{path}:{environment['PATH']}"
    environment.update(env or {})
    return subprocess.run(["/bin/sh", str(SCRIPT), *args], env=environment,
                          capture_output=True, text=True, check=False)


def test_the_script_is_executable_and_shipped_as_a_script() -> None:
    """Installed to /usr/local/bin by setuptools `scripts`, so it rides the
    same pin as the Python code and TLS logic can never be at a different
    version from auth logic inside one image."""
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_no_tls_configured_execs_the_command_untouched(fake_uvicorn: Path) -> None:
    """TLS is opt-in. An image with neither variable set behaves exactly as it
    did before this script existed."""
    result = run("uvicorn", "app.main:app", path=fake_uvicorn)
    assert result.returncode == 0
    assert result.stdout.strip() == "ARGS: app.main:app"


def test_half_a_tls_configuration_exits_rather_than_serving_plain_http(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    """stt-stack warned and served plain HTTP here.

    Everything works and nothing is encrypted, which is invisible to the
    operator reading "TLS is configured" from their own compose file while a
    bearer token crosses the LAN in the clear on every request.
    """
    cert = tmp_path / "cert.pem"
    cert.write_text("x")
    result = run("uvicorn", env={"TTS_TLS_CERT": str(cert)}, path=fake_uvicorn)
    assert result.returncode == 1
    assert "must both" in result.stderr
    assert "ARGS:" not in result.stdout


def test_only_the_key_is_also_half_a_configuration(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    key = tmp_path / "key.pem"
    key.write_text("x")
    result = run("uvicorn", env={"TTS_TLS_KEY": str(key)}, path=fake_uvicorn)
    assert result.returncode == 1


def test_tls_with_a_command_that_is_not_uvicorn_exits(tmp_path: Path) -> None:
    """tts-stack had no command check at all and appended --ssl-certfile to
    whatever CMD it was given, which breaks the CI import check that runs
    `python -c` against the same image."""
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    result = run("python", "-c", "pass",
                 env={"TTS_TLS_CERT": str(cert), "TTS_TLS_KEY": str(key)})
    assert result.returncode == 1
    assert "not" in result.stderr and "uvicorn" in result.stderr


def test_the_command_check_lives_inside_the_tls_block_not_around_it(
        fake_uvicorn: Path) -> None:
    """Gating the whole block on the command meant a compose `command:`
    override skipped every line of it: fully configured TLS, plain HTTP, and
    not one word on stderr. With no TLS configured, any command still runs."""
    result = run("python3", "-c", "print('ran')")
    assert result.returncode == 0
    assert "ran" in result.stdout


def test_a_fully_configured_tls_run_appends_uvicorns_own_flags(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    """The paths become uvicorn flags rather than being read in Python, so the
    CMD baked into the image stays exactly what it was for anyone already
    running it."""
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    result = run("uvicorn", "app.main:app",
                 env={"TTS_TLS_CERT": str(cert), "TTS_TLS_KEY": str(key)},
                 path=fake_uvicorn)
    assert result.returncode == 0
    assert result.stdout.strip().endswith(
        f"app.main:app --ssl-certfile {cert} --ssl-keyfile {key}")
    assert "serving HTTPS" in result.stdout


def test_an_unreadable_certificate_exits_rather_than_reaching_uvicorn(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    """uvicorn's failure at that point is a traceback rather than a sentence."""
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    key.chmod(0o000)
    result = run("uvicorn", env={"TTS_TLS_CERT": str(cert),
                                 "TTS_TLS_KEY": str(key)}, path=fake_uvicorn)
    key.chmod(0o600)
    assert result.returncode == 1
    assert "not readable" in result.stderr


def test_the_tls_prefix_is_a_parameter_so_no_variable_had_to_be_renamed(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    """stt-stack uses STT_TLS_*, the two tts services use TTS_TLS_*. Forcing a
    rename would be a breaking change for a deployment that did nothing."""
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    result = run("uvicorn", env={"VOICE_TLS_PREFIX": "STT",
                                 "STT_TLS_CERT": str(cert),
                                 "STT_TLS_KEY": str(key)}, path=fake_uvicorn)
    assert result.returncode == 0
    assert f"--ssl-certfile {cert}" in result.stdout


def test_the_prefix_only_reads_its_own_variables(
        tmp_path: Path, fake_uvicorn: Path) -> None:
    """Under STT, a stray TTS_TLS_CERT must not half-configure anything."""
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    result = run("uvicorn", env={"VOICE_TLS_PREFIX": "STT",
                                 "TTS_TLS_CERT": str(cert)}, path=fake_uvicorn)
    assert result.returncode == 0
    assert "--ssl-certfile" not in result.stdout


def test_nothing_generates_a_certificate() -> None:
    """A certificate that appears by magic is one every client is taught to
    stop validating, and a client taught to skip verification keeps skipping
    it against the real certificate too. All three copies already say so."""
    # Comments stripped first: the reasoning above says "self-signed" several
    # times and only the executable lines are the promise.
    code = "\n".join(line for line in SCRIPT.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    for forbidden in ("openssl", "x509", "certtool", "mkcert"):
        assert forbidden not in code.lower(), forbidden


def test_the_chown_list_is_a_parameter_because_tts_long_mounts_two_volumes(
        ) -> None:
    """/models for two services, /models /output for tts-long. Reading it as a
    space-separated list is the whole of that difference."""
    source = SCRIPT.read_text()
    assert 'VOICE_CHOWN_DIRS' in source
    assert 'for d in $CHOWN_DIRS' in source


def test_the_script_runs_under_plain_sh_not_bash() -> None:
    """The images are python:3.x-slim-trixie and have no bash. An indirect
    expansion or a [[ would work on a developer's machine and fail in the
    image, which is the worst place to find out."""
    result = subprocess.run(["/bin/sh", "-n", str(SCRIPT)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    if shutil.which("dash"):
        result = subprocess.run(["dash", "-n", str(SCRIPT)],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
