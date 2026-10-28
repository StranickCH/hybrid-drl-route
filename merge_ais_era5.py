"""
merge_ais_era5.py
-----------------
Matches the Houston AIS .parquet files with ERA5 meteorological data.
ERA5 is stored in two files:
  era5_wind_2024.nc  — wind (u10, v10)
  era5_wave_2024.nc  — waves (swh)

Output: ais_merged/AIS_YYYY_MM_DD_era5.parquet
Every AIS row gets additional columns:
  wind_u, wind_v, wind_speed, wind_dir, wave_hs, wave_risk

Usage:
  python merge_ais_era5.py
  python merge_ais_era5.py --ais-dir ais_filtered
  python merge_ais_era5.py --check
"""

import os
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ─────────────────────────── configuration ───────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

DEFAULT_AIS_DIR   = os.path.join(ROOT, "ais_filtered")
DEFAULT_ERA5_WIND = os.path.join(ROOT, "era5", "era5_wind_2024.nc")
DEFAULT_ERA5_WAVE = os.path.join(ROOT, "era5", "era5_wave_2024.nc")
DEFAULT_OUT_DIR   = os.path.join(ROOT, "ais_merged")

WAVE_RISK_THRESHOLD = 2.5  # metres - "storm" threshold

# ────────────────────────────── logging ──────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. Loading ERA5
# ═══════════════════════════════════════════════════════════════════

def load_era5(wind_path: str, wave_path: str) -> xr.Dataset:
    """
    Opens the two ERA5 NetCDF files and merges them.
    Returns an xr.Dataset with coordinates: valid_time (or time), latitude, longitude.
    """
    log.info(f"Loading ERA5 wind: {wind_path}")
    ds_wind = xr.open_dataset(wind_path, engine="netcdf4")

    log.info(f"Loading ERA5 waves: {wave_path}")
    ds_wave = xr.open_dataset(wave_path, engine="netcdf4")

    log.info(f"Wind variables: {list(ds_wind.data_vars)}")
    log.info(f"Wave variables: {list(ds_wave.data_vars)}")
    log.info(f"Wind coordinates: {list(ds_wind.coords)}")

    # Merge first, then normalise the time name
    ds = xr.merge([ds_wind, ds_wave], join="inner")

    # ERA5 sometimes uses 'valid_time' instead of 'time' - fixed after the merge
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    time_coord = "time" if "time" in ds.coords else list(ds.coords)[0]
    log.info(f"ERA5 merged: {len(ds[time_coord])} time steps, variables: {list(ds.data_vars)}")
    log.info(f"Time coordinate: '{time_coord}'")

    return ds


# ═══════════════════════════════════════════════════════════════════
#  2. ERA5 interpolation for one AIS file
# ═══════════════════════════════════════════════════════════════════

def find_era5_var(ds: xr.Dataset, candidates: list) -> str | None:
    """Returns the first candidate name found in the dataset."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def merge_file(ais_path: Path, ds: xr.Dataset, out_dir: Path) -> dict:
    """
    Reads one AIS parquet file and attaches ERA5 by time and position.
    Returns a dict with statistics.
    """
    out_path = out_dir / ais_path.name.replace(".parquet", "_era5.parquet")

    # Skip files that are already processed
    if out_path.exists():
        log.info(f"Skipped (already exists): {out_path.name}")
        return {"file": ais_path.name, "status": "skipped", "rows": 0}

    try:
        df = pd.read_parquet(ais_path)
    except Exception as e:
        log.error(f"Error reading {ais_path.name}: {e}")
        return {"file": ais_path.name, "status": "read_error", "rows": 0}

    if df.empty:
        return {"file": ais_path.name, "status": "empty", "rows": 0}

    # ── Find the time and position columns ──
    time_col = next((c for c in ["timestamp", "BaseDateTime", "time", "datetime"] if c in df.columns), None)
    lat_col  = next((c for c in ["LAT", "lat", "latitude"] if c in df.columns), None)
    lon_col  = next((c for c in ["LON", "lon", "longitude"] if c in df.columns), None)

    if not all([time_col, lat_col, lon_col]):
        log.error(f"Required columns missing in {ais_path.name}. Present: {list(df.columns)}")
        return {"file": ais_path.name, "status": "col_error", "rows": 0}

    # ── Convert types ──
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col])

    # ── Nearest ERA5 hour ──
    df["era5_time"] = df[time_col].dt.round("1h")

    # ── Find the variable names ──
    u_var  = find_era5_var(ds, ["u10", "U10", "u_component_of_wind"])
    v_var  = find_era5_var(ds, ["v10", "V10", "v_component_of_wind"])
    hs_var = find_era5_var(ds, ["swh", "SWH", "significant_height_of_combined_wind_waves_and_swell"])

    if not u_var or not v_var:
        log.error(f"No wind variables in ERA5. Available: {list(ds.data_vars)}")
        return {"file": ais_path.name, "status": "era5_var_error", "rows": 0}

    # ── ERA5: time range ──
    # ERA5 times without timezone for compatibility with the AIS data
    # (some AIS files have naive datetimes, others UTC-aware)
    era5_times_utc = pd.DatetimeIndex(ds.time.values)
    if era5_times_utc.tz is not None:
        era5_times = era5_times_utc.tz_localize(None)
    else:
        era5_times = era5_times_utc
    t_min, t_max = era5_times.min(), era5_times.max()

    # Make era5_time in df naive as well
    if df["era5_time"].dt.tz is not None:
        df["era5_time"] = df["era5_time"].dt.tz_localize(None)

    mask = (df["era5_time"] >= t_min) & (df["era5_time"] <= t_max)
    df_ok  = df[mask].copy()
    df_out = df[~mask].copy()

    # ── Interpolation over unique (era5_time, lat, lon) ──
    # FIX: previously drop_duplicates("era5_time") took the position of one
    # random vessel for the whole time step. Now ERA5 is extracted
    # for every unique combination (era5_time, rounded position).
    if not df_ok.empty:
        # Round the position to the nearest ERA5 node (0.25 degree)
        df_ok["era5_lat"] = (df_ok[lat_col] / 0.25).round() * 0.25
        df_ok["era5_lon"] = (df_ok[lon_col] / 0.25).round() * 0.25

        pts = df_ok[["era5_time", "era5_lat", "era5_lon"]].drop_duplicates().copy()

        results = []
        for _, row in pts.iterrows():
            t  = row["era5_time"]
            # xarray ERA5 uses naive datetimes - drop the timezone if present
            if hasattr(t, "tzinfo") and t.tzinfo is not None:
                t = t.tz_localize(None)
            la = float(row["era5_lat"])
            lo = float(row["era5_lon"])
            try:
                sel = ds.sel(time=t, latitude=la, longitude=lo, method="nearest")
                u  = float(sel[u_var].values)
                v  = float(sel[v_var].values)
                hs = float(sel[hs_var].values) if hs_var else np.nan
                results.append({"era5_time": t, "era5_lat": la, "era5_lon": lo,
                                 "wind_u": u, "wind_v": v, "wave_hs": hs})
            except Exception:
                results.append({"era5_time": t, "era5_lat": la, "era5_lon": lo,
                                 "wind_u": np.nan, "wind_v": np.nan, "wave_hs": np.nan})

        era5_df = pd.DataFrame(results)
        df_ok = df_ok.merge(era5_df, on=["era5_time", "era5_lat", "era5_lon"], how="left")
        df_ok = df_ok.drop(columns=["era5_lat", "era5_lon"], errors="ignore")
    else:
        df_ok["wind_u"]  = np.nan
        df_ok["wind_v"]  = np.nan
        df_ok["wave_hs"] = np.nan

    if not df_out.empty:
        df_out["wind_u"]  = np.nan
        df_out["wind_v"]  = np.nan
        df_out["wave_hs"] = np.nan

    df_final = pd.concat([df_ok, df_out], ignore_index=True)

    # ── Derived variables ──
    df_final["wind_speed"] = np.sqrt(df_final["wind_u"]**2 + df_final["wind_v"]**2)
    df_final["wind_dir"]   = (np.degrees(np.arctan2(df_final["wind_u"], df_final["wind_v"])) + 360) % 360
    df_final["wave_risk"]  = (df_final["wave_hs"] > WAVE_RISK_THRESHOLD).astype(int)

    df_final = df_final.drop(columns=["era5_time"], errors="ignore")
    df_final.to_parquet(out_path, index=False)

    return {"file": ais_path.name, "status": "ok", "rows": len(df_final)}


# ═══════════════════════════════════════════════════════════════════
#  3. Quick check of the results
# ═══════════════════════════════════════════════════════════════════

def quick_check(out_dir: Path):
    files = sorted(out_dir.glob("*_era5.parquet"))
    if not files:
        log.warning(f"No *_era5.parquet files in {out_dir}")
        return

    log.info(f"\n{'='*55}")
    log.info(f"Check: {len(files)} files in {out_dir}")

    # Read the first file
    df = pd.read_parquet(files[0])
    log.info(f"First file: {files[0].name}")
    log.info(f"  Rows: {len(df):,}")
    log.info(f"  Columns: {list(df.columns)}")

    for col in ["wind_u", "wind_v", "wind_speed", "wave_hs"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            log.info(f"  {col}: {valid:,} values, mean={df[col].mean():.2f}")

    log.info(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════════════
#  4. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Merge AIS + ERA5")
    parser.add_argument("--ais-dir",   default=DEFAULT_AIS_DIR,   help="Folder with AIS .parquet files")
    parser.add_argument("--era5-wind", default=DEFAULT_ERA5_WIND, help="ERA5 wind file (.nc)")
    parser.add_argument("--era5-wave", default=DEFAULT_ERA5_WAVE, help="ERA5 wave file (.nc)")
    parser.add_argument("--out-dir",   default=DEFAULT_OUT_DIR,   help="Output folder")
    parser.add_argument("--check",     action="store_true",        help="Only check the finished files")
    parser.add_argument("--reprocess-months", type=int, nargs="+", default=None,
                        help="Rewrite the files of the given months only (e.g. --reprocess-months 5 6)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        quick_check(out_dir)
        return

    # Load ERA5 once
    ds = load_era5(args.era5_wind, args.era5_wave)

    # Find all AIS parquet files (top level of the folder only)
    ais_dir = Path(args.ais_dir)
    parquet_files = sorted([
        f for f in ais_dir.glob("*.parquet")
        if not f.name.startswith("_")
    ])

    log.info(f"AIS files found: {len(parquet_files)}")

    if not parquet_files:
        log.error(f"No .parquet files in {ais_dir}")
        return

    # Filter by month if --reprocess-months is given
    if args.reprocess_months:
        def file_month(p):
            # Name like AIS_2024_05_01.parquet
            parts = p.stem.split("_")
            try:
                return int(parts[2])  # month at position 2
            except (IndexError, ValueError):
                return None
        parquet_files = [p for p in parquet_files if file_month(p) in args.reprocess_months]
        log.info(f"Rewriting months {args.reprocess_months} only: {len(parquet_files)} files")
        # Delete the old output files so that merge_file does not skip them
        for pf in parquet_files:
            out_f = out_dir / pf.name.replace(".parquet", "_era5.parquet")
            if out_f.exists():
                out_f.unlink()
                log.info(f"  Deleted old file: {out_f.name}")

    # Process every file
    summary = []
    t0 = time.time()
    for pf in tqdm(parquet_files, desc="Merge AIS+ERA5", unit="file"):
        result = merge_file(pf, ds, out_dir)
        summary.append(result)

    # Summary
    elapsed = time.time() - t0
    summary_df = pd.DataFrame(summary)
    print("\n" + "="*60)
    print("TOTAL:")
    ok_df = summary_df[summary_df["status"] == "ok"]
    print(f"  Processed:   {len(ok_df)} files")
    print(f"  Skipped:     {(summary_df['status'] == 'skipped').sum()} files")
    print(f"  Errors:      {(~summary_df['status'].isin(['ok','skipped'])).sum()} files")
    print(f"  Total rows:  {ok_df['rows'].sum():,}")
    print(f"  Time:        {elapsed/60:.1f} min")
    print("="*60)

    summary_path = out_dir / "_merge_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary: {summary_path}")

    quick_check(out_dir)


if __name__ == "__main__":
    main()
