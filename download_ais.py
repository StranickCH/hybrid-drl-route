"""
download_ais.py
===============
Download daily AIS files of Marine Cadastre (NOAA) from coast.noaa.gov.

Source: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/YYYY/
File format: AIS_YYYY_MM_DD.zip  (contains AIS_YYYY_MM_DD.csv)
CSV columns: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName,
             IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo,
             TransceiverClass

Usage (Windows, PowerShell):
    python download_ais.py --start 2024-01-01 --end 2024-01-31 --out ais_raw
    python download_ais.py --start 2024-06-01 --end 2024-06-07 --workers 4
    python download_ais.py --start 2025-01-01 --end 2025-01-31 --dry-run

Features:
 - URLs are built directly from the file name pattern, no listing is needed.
 - Resumable: existing files are skipped.
 - Incomplete downloads (.part) are restarted from zero.
 - 404 (date not available) does not stop the run.
 - Parallel downloads (4 threads by default).

Dependencies: Python >= 3.9 standard library only.
Optional: pip install tqdm
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm  # type: ignore
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
CHUNK = 1 << 20  # 1 MB


def build_daily_index(start: date, end: date) -> dict[date, dict]:
    """
    Builds the URL index directly from the file name pattern.
    Files: AIS_YYYY_MM_DD.zip at coast.noaa.gov/htdata/CMSP/AISDataHandler/YYYY/
    """
    idx: dict[date, dict] = {}
    d = start
    while d <= end:
        fname = f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"
        idx[d] = {
            "name": fname,
            "url": f"{BASE_URL}/{d.year}/{fname}",
            "size": 0,
        }
        d += timedelta(days=1)
    return idx


def download_one(url: str, target: Path, expected_size: int = 0,
                 retries: int = 3, timeout: int = 300) -> tuple[Path, bool, str]:
    """
    Downloads a file to target. Skips it if it already exists and is not empty.
    Uses a .part file for atomicity.
    """
    if target.exists() and target.stat().st_size > 0:
        if expected_size and target.stat().st_size != expected_size:
            target.unlink(missing_ok=True)
        else:
            return (target, True, f"skip ({target.stat().st_size / 1e6:.1f} MB)")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ais-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
            if expected_size and tmp.stat().st_size != expected_size:
                last_err = f"size mismatch: {tmp.stat().st_size} != {expected_size}"
                tmp.unlink(missing_ok=True)
                time.sleep(2 * attempt)
                continue
            tmp.replace(target)
            return (target, True, f"ok ({target.stat().st_size / 1e6:.1f} MB)")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                tmp.unlink(missing_ok=True)
                return (target, False, "404 (no file)")
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)[:140]
        time.sleep(2 * attempt)

    tmp.unlink(missing_ok=True)
    return (target, False, f"failed after {retries}: {last_err}")


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run(start: date, end: date, out_root: Path, workers: int = 4,
        dry_run: bool = False) -> None:

    print(f"[i] building file list {start} ... {end} ...", flush=True)
    index = build_daily_index(start, end)
    print(f"[i] to download: {len(index)} dates")

    tasks = [(d, blob, out_root / str(d.year) / blob["name"])
             for d, blob in index.items()]

    if dry_run:
        for d, blob, target in tasks:
            print(f"  {d.isoformat()}  -> {blob['url']}")
        return

    ok = skipped = fail = not_found = 0
    progress = tqdm(total=len(tasks), unit="file", desc="AIS") if HAVE_TQDM else None

    def _worker(args):
        d, blob, target = args
        return (d, *download_one(blob["url"], target, expected_size=blob["size"]))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for d, target, success, msg in ex.map(_worker, tasks):
            if success:
                if msg.startswith("skip"):
                    skipped += 1
                else:
                    ok += 1
            else:
                if "404" in msg:
                    not_found += 1
                else:
                    fail += 1
            line = f"{d.isoformat()}  {'OK ' if success else 'ERR'}  {msg}  {target.name}"
            if progress is not None:
                progress.update(1)
                progress.write(line)
            else:
                print(line, flush=True)

    if progress is not None:
        progress.close()

    print(f"\n[total] downloaded={ok}  skipped={skipped}  "
          f"errors={fail}  404_no_file={not_found}")


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}") from e


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download daily AIS ZIP files of Marine Cadastre (NOAA)",
    )
    ap.add_argument("--start", required=True, type=parse_date)
    ap.add_argument("--end",   required=True, type=parse_date)
    ap.add_argument("--out",   default=str(Path(__file__).resolve().parent / "ais_raw"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.end < args.start:
        ap.error("--end is earlier than --start")

    run(args.start, args.end, Path(args.out), workers=args.workers, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
