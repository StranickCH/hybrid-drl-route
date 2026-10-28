"""
filter_ais.py
=============
Filters raw daily AIS files (*.zip or *.csv.gz) by bounding box
and vessel type and converts them to Parquet.

Typical pipeline:
    # 1) download
    python download_ais.py --start 2024-01-01 --end 2024-01-31

    # 2) filter: Houston + cargo ships and tankers
    python filter_ais.py --in-dir ais_raw --out-dir ais_filtered --start 2024-01-01 --end 2024-01-31 --bbox 28.5,-95.6,30.0,-94.0 --vessel-types 70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89

bbox format: lat_min,lon_min,lat_max,lon_max
VesselType codes:
    70-79 — Cargo,  80-89 — Tanker,  60-69 — Passenger

Dependencies: pip install pandas pyarrow
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:
    print("[!] pandas is required: pip install pandas pyarrow", file=sys.stderr)
    raise


AIS_DTYPES = {
    "MMSI": "int64",
    "LAT": "float32",
    "LON": "float32",
    "SOG": "float32",
    "COG": "float32",
    "Heading": "float32",
    "VesselName": "string",
    "IMO": "string",
    "CallSign": "string",
    "VesselType": "Int16",
    "Status": "Int16",
    "Length": "float32",
    "Width": "float32",
    "Draft": "float32",
    "Cargo": "Int16",
    "TransceiverClass": "string",
}
PARSE_DATES = ["BaseDateTime"]


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox: lat_min,lon_min,lat_max,lon_max")
    lat_min, lon_min, lat_max, lon_max = map(float, parts)
    if not (-90 <= lat_min < lat_max <= 90 and -180 <= lon_min < lon_max <= 180):
        raise argparse.ArgumentTypeError("invalid bbox")
    return lat_min, lon_min, lat_max, lon_max


def parse_types(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def locate_input(in_dir: Path, d: date) -> Path | None:
    """Looks for AIS_YYYY_MM_DD.zip or .csv.gz in in_dir (including year subfolders)."""
    stem = f"AIS_{d.year}_{d.month:02d}_{d.day:02d}"
    for ext in (".zip", ".csv.gz"):
        for base in (in_dir / str(d.year), in_dir):
            p = base / (stem + ext)
            if p.exists():
                return p
    # fallback recursive search
    for ext in (".zip", ".csv.gz"):
        for p in in_dir.rglob(stem + ext):
            return p
    return None


def _compression(path: Path) -> str:
    """Detects the compression type from the file extension."""
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".csv.gz") or name.endswith(".gz"):
        return "gzip"
    return "infer"


def process_day_pandas(src: Path, bbox, vessel_types,
                       chunksize: int = 1_000_000) -> "pd.DataFrame":
    """
    Reads a daily ZIP/CSV.GZ as a stream, filters it and returns a DataFrame.
    Works without running out of memory on ~1-3 GB files.
    """
    usecols = list(AIS_DTYPES.keys()) + PARSE_DATES
    compression = _compression(src)
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        src,
        compression=compression,
        chunksize=chunksize,
        usecols=lambda c: c in usecols or c in PARSE_DATES,
        dtype={k: v for k, v in AIS_DTYPES.items()},
        parse_dates=PARSE_DATES,
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["LAT", "LON", "BaseDateTime", "MMSI"])
        if bbox is not None:
            lat_min, lon_min, lat_max, lon_max = bbox
            m = (
                chunk["LAT"].between(lat_min, lat_max, inclusive="both")
                & chunk["LON"].between(lon_min, lon_max, inclusive="both")
            )
            chunk = chunk.loc[m]
        if vessel_types:
            chunk = chunk.loc[chunk["VesselType"].isin(vessel_types)]
        if "SOG" in chunk.columns:
            chunk = chunk.loc[chunk["SOG"].between(0, 60) | chunk["SOG"].isna()]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def process_day_polars(src: Path, bbox, vessel_types):
    """Faster alternative using polars."""
    import polars as pl  # type: ignore
    lf = pl.scan_csv(src, try_parse_dates=True, infer_schema_length=10_000,
                     ignore_errors=True)
    if bbox is not None:
        lat_min, lon_min, lat_max, lon_max = bbox
        lf = lf.filter(
            (pl.col("LAT") >= lat_min) & (pl.col("LAT") <= lat_max)
            & (pl.col("LON") >= lon_min) & (pl.col("LON") <= lon_max)
        )
    if vessel_types:
        lf = lf.filter(pl.col("VesselType").is_in(vessel_types))
    lf = lf.filter(pl.col("SOG").is_between(0, 60) | pl.col("SOG").is_null())
    return lf.collect().to_pandas()


def run(args) -> int:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t0 = time.time()

    for d in daterange(args.start, args.end):
        src = locate_input(in_dir, d)
        if src is None:
            print(f"[{d.isoformat()}] source not found - skipped")
            summary_rows.append({"date": d.isoformat(), "status": "missing",
                                  "rows": 0, "size_mb": 0.0})
            continue

        dst = out_dir / f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.parquet"
        if dst.exists() and not args.overwrite:
            print(f"[{d.isoformat()}] {dst.name} already exists - skipped (--overwrite to overwrite)")
            summary_rows.append({"date": d.isoformat(), "status": "skipped",
                                  "rows": -1, "size_mb": dst.stat().st_size / 1e6})
            continue

        td = time.time()
        try:
            if args.engine == "polars":
                df = process_day_polars(src, args.bbox, args.vessel_types)
            else:
                df = process_day_pandas(src, args.bbox, args.vessel_types,
                                        chunksize=args.chunksize)
        except Exception as e:
            print(f"[{d.isoformat()}] ERROR: {e}")
            summary_rows.append({"date": d.isoformat(), "status": f"error: {e}",
                                  "rows": 0, "size_mb": 0.0})
            continue

        if df.empty:
            print(f"[{d.isoformat()}] 0 rows after filtering")
            summary_rows.append({"date": d.isoformat(), "status": "empty",
                                  "rows": 0, "size_mb": 0.0})
            continue

        df = df.sort_values(["MMSI", "BaseDateTime"], kind="mergesort").reset_index(drop=True)
        df.to_parquet(dst, index=False, compression="zstd")
        size_mb = dst.stat().st_size / 1e6
        print(f"[{d.isoformat()}] {len(df):>9,} rows  "
              f"{size_mb:6.1f} MB  in {time.time() - td:5.1f} s  -> {dst.name}")
        summary_rows.append({"date": d.isoformat(), "status": "ok",
                              "rows": len(df), "size_mb": size_mb})

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    total_rows = int(summary.loc[summary["status"] == "ok", "rows"].sum())
    total_mb = float(summary["size_mb"].sum())
    print(f"\n[total] rows={total_rows:,}  Parquet size={total_mb:.1f} MB  "
          f"time={time.time() - t0:.1f} s  summary -> {summary_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter AIS ZIP/CSV.GZ -> Parquet")
    ap.add_argument("--in-dir",  default=str(Path(__file__).resolve().parent / "ais_raw"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", required=True, type=parse_date)
    ap.add_argument("--end",   required=True, type=parse_date)
    ap.add_argument("--bbox",  type=parse_bbox, default=None)
    ap.add_argument("--vessel-types", type=parse_types, default=[])
    ap.add_argument("--engine", choices=("pandas", "polars"), default="pandas")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.end < args.start:
        ap.error("--end is earlier than --start")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
