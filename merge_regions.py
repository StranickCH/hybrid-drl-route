"""
merge_regions.py  v2
--------------------------
AIS + ERA5 merge for the 4 additional regions (Chesapeake Bay, New York Harbor, Mississippi Delta, Puget Sound).

Details:
  - ERA5 valid_time is used directly (no rename)
  - AIS times are converted to naive (no UTC) for the comparison
  - Files that are already done are skipped

Usage: python merge_regions.py
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WAVE_RISK = 2.5

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

REGIONS = [
    {
        "name":      "Chesapeake",
        "ais_dir":   os.path.join(ROOT, "ais_chesapeake"),
        "era5_wind": os.path.join(ROOT, "era5_chesapeake", "era5_wind_2024.nc"),
        "era5_wave": os.path.join(ROOT, "era5_chesapeake", "era5_wave_2024.nc"),
        "out_dir":   os.path.join(ROOT, "ais_merged_chesapeake"),
    },
    {
        "name":      "NewYork",
        "ais_dir":   os.path.join(ROOT, "ais_newyork"),
        "era5_wind": os.path.join(ROOT, "era5_newyork", "era5_wind_2024.nc"),
        "era5_wave": os.path.join(ROOT, "era5_newyork", "era5_wave_2024.nc"),
        "out_dir":   os.path.join(ROOT, "ais_merged_newyork"),
    },
    {
        "name":      "Mississippi",
        "ais_dir":   os.path.join(ROOT, "ais_mississippi"),
        "era5_wind": os.path.join(ROOT, "era5_mississippi", "era5_wind_2024.nc"),
        "era5_wave": os.path.join(ROOT, "era5_mississippi", "era5_wave_2024.nc"),
        "out_dir":   os.path.join(ROOT, "ais_merged_mississippi"),
    },
    {
        "name":      "PugetSound",
        "ais_dir":   os.path.join(ROOT, "ais_pugetsound"),
        "era5_wind": os.path.join(ROOT, "era5_pugetsound", "era5_wind_2024.nc"),
        "era5_wave": os.path.join(ROOT, "era5_pugetsound", "era5_wave_2024.nc"),
        "out_dir":   os.path.join(ROOT, "ais_merged_pugetsound"),
    },
]


def load_era5(wind_path, wave_path):
    ds_wind = xr.open_dataset(wind_path, engine="netcdf4")
    ds_wave = xr.open_dataset(wave_path, engine="netcdf4")
    ds = xr.merge([ds_wind, ds_wave], join="inner")

    # Find the name of the time coordinate
    time_dim = next((c for c in ["valid_time", "time"] if c in ds.coords), None)

    # ERA5 times → naive (no timezone) for a uniform comparison
    era5_times_raw = pd.to_datetime(ds[time_dim].values)
    if era5_times_raw.tz is not None:
        era5_times = era5_times_raw.tz_localize(None)
    else:
        era5_times = era5_times_raw

    log.info(f"  ERA5: {time_dim}, {len(era5_times)} steps")
    log.info(f"  Variables: {list(ds.data_vars)}")
    return ds, time_dim, era5_times


def to_naive(series):
    """Converts a Series of times to naive datetimes (drops the timezone)."""
    s = pd.to_datetime(series, errors="coerce")
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s


def merge_file(f, ds, time_dim, era5_times, out_dir):
    out_path = Path(out_dir) / f.name.replace(".parquet", "_era5.parquet")
    if out_path.exists():
        return "skipped"

    df = pd.read_parquet(f)
    if df.empty:
        return "empty"

    time_col = next((c for c in ["BaseDateTime","timestamp","time"] if c in df.columns), None)
    lat_col  = next((c for c in ["LAT","lat","latitude"] if c in df.columns), None)
    lon_col  = next((c for c in ["LON","lon","longitude"] if c in df.columns), None)
    if not all([time_col, lat_col, lon_col]):
        return "col_error"

    # Time → naive
    df[time_col] = to_naive(df[time_col])
    df = df.dropna(subset=[time_col, lat_col, lon_col])

    # Round to 6 h (ERA5 step)
    df["era5_time"] = df[time_col].dt.round("6h")

    t_min, t_max = era5_times.min(), era5_times.max()
    mask   = df["era5_time"].between(t_min, t_max)
    df_ok  = df[mask].copy()
    df_out = df[~mask].copy()

    u_var  = next((v for v in ["u10","U10"] if v in ds.data_vars), None)
    v_var  = next((v for v in ["v10","V10"] if v in ds.data_vars), None)
    hs_var = next((v for v in ["swh","SWH"] if v in ds.data_vars), None)

    rows = []
    if not df_ok.empty and u_var:
        for t in df_ok["era5_time"].unique():
            sub = df_ok[df_ok["era5_time"] == t]
            la  = float(sub[lat_col].median())
            lo  = float(sub[lon_col].median())
            try:
                sel = ds.sel({time_dim: t, "latitude": la, "longitude": lo},
                             method="nearest")
                u  = float(sel[u_var].values)
                v  = float(sel[v_var].values) if v_var else np.nan
                hs = float(sel[hs_var].values) if hs_var else np.nan
            except Exception:
                u, v, hs = np.nan, np.nan, np.nan
            rows.append({"era5_time": t, "wind_u": u, "wind_v": v, "wave_hs": hs})

    if rows:
        era5_df = pd.DataFrame(rows)
        df_ok = df_ok.merge(era5_df, on="era5_time", how="left")
    else:
        df_ok["wind_u"] = df_ok["wind_v"] = df_ok["wave_hs"] = np.nan

    for col in ["wind_u","wind_v","wave_hs"]:
        df_out[col] = np.nan

    df_final = pd.concat([df_ok, df_out], ignore_index=True)
    df_final["wind_speed"] = np.sqrt(df_final["wind_u"]**2 + df_final["wind_v"]**2)
    df_final["wind_dir"]   = (np.degrees(
        np.arctan2(df_final["wind_u"], df_final["wind_v"])) + 360) % 360
    df_final["wave_risk"]  = (df_final["wave_hs"] > WAVE_RISK).astype(int)
    df_final = df_final.drop(columns=["era5_time"], errors="ignore")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_path, index=False)

    valid = df_final["wind_u"].notna().sum()
    log.info(f"    {f.name}: wind_u {valid:,}/{len(df_final):,} rows")
    return "ok"


def merge_region(region):
    name    = region["name"]
    out_dir = region["out_dir"]

    # Check how many are already done
    existing = list(Path(out_dir).glob("*_era5.parquet")) if Path(out_dir).exists() else []
    files    = sorted(Path(region["ais_dir"]).glob("*.parquet"))

    if len(existing) >= len(files) and files:
        log.info(f"  {name}: already done ({len(existing)} files) - skipped")
        return

    log.info(f"\n{'='*55}")
    log.info(f"  Region: {name}  ({len(files)} AIS files)")
    log.info(f"{'='*55}")

    if not files:
        log.warning(f"  No AIS parquet in {region['ais_dir']}")
        return

    ds, time_dim, era5_times = load_era5(region["era5_wind"], region["era5_wave"])

    t0 = time.time()
    for f in tqdm(files, desc=f"Merge {name}"):
        merge_file(f, ds, time_dim, era5_times, out_dir)

    # Final check
    merged = list(Path(out_dir).glob("*_era5.parquet"))
    log.info(f"  ✅ {name}: {len(merged)} files [{time.time()-t0:.0f}s]")
    if merged:
        df_check = pd.read_parquet(merged[0])
        for col in ["wind_u","wind_v","wind_speed","wave_hs"]:
            if col in df_check.columns:
                v = df_check[col].notna().sum()
                m = df_check[col].mean()
                log.info(f"    {col}: {v:,} values, mean={m:.3f}")


def main():
    for region in REGIONS:
        merge_region(region)
    log.info("\n✅ All regions done!")


if __name__ == "__main__":
    main()
