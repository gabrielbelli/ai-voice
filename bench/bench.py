#!/usr/bin/env python3
"""Benchmark stt-stack across locales, sources and recording conditions.

    python bench.py fetch                      cache samples locally
    python bench.py run  --url http://host:8000
    python bench.py report                     compare runs

Fetching is separate from running so every run uses byte-identical audio. A
benchmark whose inputs move cannot tell you whether a change helped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import degrade  # noqa: E402
from sources import SOURCES  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RUNS = HERE / "runs"
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------- fetch ----

def _decode_row(entry: dict) -> np.ndarray | None:
    """Decode one audio cell to 16 kHz mono float32.

    datasets 4.x decodes audio through torchcodec, so `Audio(decode=False)`
    hands back raw bytes instead and the decoding happens here. That keeps the
    torch stack out of a benchmark client whose only other job is HTTP.
    """
    import io
    import soundfile as sf
    import soxr

    data = entry.get("bytes")
    src = io.BytesIO(data) if data else entry.get("path")
    if src is None:
        return None
    try:
        x, rate = sf.read(src, dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if rate != SAMPLE_RATE:
        x = soxr.resample(x, rate, SAMPLE_RATE)
    return x.astype(np.float32)


def fetch(per_source: int, seed: int, only: list[str]) -> None:
    from datasets import Audio, load_dataset
    import soundfile as sf

    CACHE.mkdir(exist_ok=True)
    for locale, srcs in SOURCES.items():
        if only and locale not in only:
            continue
        for src in srcs:
            out = CACHE / locale / src["id"]
            if (out / "manifest.json").is_file():
                print(f"  {locale}/{src['id']}: cached")
                continue
            out.mkdir(parents=True, exist_ok=True)
            print(f"  {locale}/{src['id']}: fetching ...", flush=True)
            try:
                ds = load_dataset(src["dataset"], src["config"],
                                  split=src["split"], streaming=True)
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest
                print(f"    SKIPPED: {exc}")
                continue

            rows, seen = [], 0
            for row in ds:
                if seen > per_source * 40:
                    break
                seen += 1
                # Corpora disagree on the field name: Common Voice uses
                # "accents", VCTK uses "accent". Checking only one silently
                # filters everything out and yields an empty, valid-looking
                # manifest.
                accent = src.get("accent")
                if accent:
                    field = str(row.get("accents") or row.get("accent") or "")
                    if accent not in field:
                        continue
                text = (row.get(src["text_key"]) or "").strip()
                if not text:
                    continue
                a = _decode_row(row["audio"])
                if a is None:
                    continue
                # Very short clips measure endpointing, not recognition; very
                # long ones dominate the wall clock for no extra information.
                if not (1.0 <= a.size / SAMPLE_RATE <= 20.0):
                    continue
                name = f"{len(rows):04d}.wav"
                sf.write(out / name, a, SAMPLE_RATE)
                rows.append({"file": name, "text": text})
                if len(rows) >= per_source:
                    break

            json.dump({"source": src, "seed": seed, "rows": rows},
                      open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)
            print(f"    {len(rows)} clips")


# ------------------------------------------------------------------ run ----

def normalise(text: str, locale: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Whisper emits cased and punctuated text; most references are neither.
    Skipping this measures formatting, not recognition. Accents are KEPT for
    pt-BR — stripping them would hide a real class of error.
    """
    import re
    import unicodedata
    t = text.lower()
    t = re.sub(r"[^\w\s'’-]", " ", t, flags=re.UNICODE)
    if not locale.startswith("pt"):
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


def run(url: str, label: str, conditions: list[str], limit: int | None,
        only: list[str], engine: str = "http", cli: str = "",
        vocab: str = "") -> None:
    import requests
    import soundfile as sf
    from jiwer import cer, wer

    RUNS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    results = []

    # The benchmark is a client like any other. Point it at a service with
    # STT_API_KEYS set and every request comes back 401 unless STT_API_KEY is
    # exported here.
    key = os.getenv("STT_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    local = None
    if engine == "parakeet":
        from parakeet_local import ParakeetLocal
        local = ParakeetLocal(cli, vocab or None)
    elif engine == "whisper-local":
        from whisper_local import WhisperLocal
        local = WhisperLocal(threads=6, hotwords=None)

    for locale in sorted(SOURCES):
        if only and locale not in only:
            continue
        for sdir in sorted((CACHE / locale).glob("*")) if (CACHE / locale).is_dir() else []:
            man_path = sdir / "manifest.json"
            if not man_path.is_file():
                continue
            man = json.load(open(man_path))
            print(f"  {locale}/{sdir.name}: starting", flush=True)
            rows = man["rows"][:limit] if limit else man["rows"]
            pool = [sf.read(sdir / r["file"], dtype="float32")[0] for r in rows[:8]]

            for cond, spec in degrade.MATRIX:
                if conditions and cond not in conditions:
                    continue
                if "codec" in spec and not degrade.HAVE_FFMPEG:
                    print(f"  {locale}/{sdir.name}/{cond}: SKIPPED (no ffmpeg)")
                    continue

                refs, hyps, rtfs = [], [], []
                for r in rows:
                    x, _ = sf.read(sdir / r["file"], dtype="float32")
                    try:
                        y = degrade.apply(x, spec, pool, rng)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    {cond}: degradation failed: {exc}")
                        break
                    import io
                    buf = io.BytesIO()
                    sf.write(buf, y, SAMPLE_RATE, format="WAV")
                    buf.seek(0)
                    try:
                        if local is not None:
                            resp = local.transcribe(y)
                        else:
                            resp = requests.post(f"{url}/transcribe",
                                                 files={"file": ("a.wav", buf, "audio/wav")},
                                                 headers=headers,
                                                 timeout=600).json()
                    except Exception as exc:  # noqa: BLE001
                        print(f"    {cond}: request failed: {exc}")
                        break
                    refs.append(normalise(r["text"], locale))
                    hyps.append(normalise(resp.get("text", ""), locale))
                    rtfs.append(resp.get("realtime_factor") or 0)

                if not refs:
                    continue
                # jiwer errors on an empty reference; an empty hypothesis is
                # fine and is exactly what a silence suite should produce.
                pairs = [(r, h) for r, h in zip(refs, hyps) if r]
                results.append({
                    "locale": locale, "source": sdir.name,
                    "style": man["source"].get("style"), "condition": cond,
                    "clips": len(pairs),
                    "wer": round(wer([r for r, _ in pairs], [h for _, h in pairs]), 4),
                    "cer": round(cer([r for r, _ in pairs], [h for _, h in pairs]), 4),
                    "rtf": round(float(np.mean(rtfs)), 2),
                })
                last = results[-1]
                print(f"  {locale:6} {sdir.name:18} {cond:16} "
                      f"WER={last['wer']:.3f} CER={last['cer']:.3f} rtf={last['rtf']}",
                      flush=True)

    out = RUNS / f"{label}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


# --------------------------------------------------------------- report ----

def report(labels: list[str]) -> None:
    runs = {}
    for p in sorted(RUNS.glob("*.json")):
        if labels and p.stem not in labels:
            continue
        runs[p.stem] = json.load(open(p))
    if not runs:
        print("no runs found")
        return

    keys = sorted({(r["locale"], r["source"], r["condition"])
                   for rs in runs.values() for r in rs})
    names = list(runs)
    print(f"{'locale':7}{'source':19}{'condition':17}" + "".join(f"{n:>12}" for n in names))
    for k in keys:
        row = f"{k[0]:7}{k[1]:19}{k[2]:17}"
        for n in names:
            m = next((r for r in runs[n] if (r["locale"], r["source"], r["condition"]) == k), None)
            row += f"{m['wer']:>12.3f}" if m else f"{'-':>12}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--per-source", type=int, default=100)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--locales", nargs="*", default=[], help="e.g. pt-BR")

    r = sub.add_parser("run")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--label", required=True, help="name this configuration")
    r.add_argument("--conditions", nargs="*", default=[],
                   help="subset of the degradation matrix; default all")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--locales", nargs="*", default=[], help="e.g. pt-BR")
    r.add_argument("--engine", choices=["http", "parakeet", "whisper-local"], default="http")
    r.add_argument("--cli", default="", help="path to fluidaudiocli")
    r.add_argument("--vocab", default="", help="custom vocab json for parakeet")

    p = sub.add_parser("report")
    p.add_argument("labels", nargs="*")

    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.per_source, a.seed, a.locales)
    elif a.cmd == "run":
        run(a.url, a.label, a.conditions, a.limit, a.locales,
            a.engine, a.cli, a.vocab)
    else:
        report(a.labels)
