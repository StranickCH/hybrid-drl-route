"""
download_era5_regions.py
------------------------
Downloads ERA5 meteorological data for all regions (Copernicus CDS API).
Houston (region 1) is the main study area; regions 2-5 are used for the cross-region tests.

Regions: Houston, Chesapeake Bay, New York Harbor,
         Mississippi Delta, Puget Sound.

Variables: wind (u10, v10) + waves (swh) - the ones used
in merge_ais_era5.py and merge_regions.py.

Usage:
  python download_era5_regions.py            # all regions
  python download_era5_regions.py --region 2 # Chesapeake only
  python download_era5_regions.py --check    # file status

Requires: pip install cdsapi
Requires: ~/.cdsapirc with your CDS API key
"""

import os
import ssl
import time
import argparse
import urllib3
from pathlib import Path

# SSL verification is disabled (needed on the authors' university network)
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

BASE_DIR = ROOT

# Months (Jan–Jun 2024)
MONTHS = ["01", "02", "03", "04", "05", "06"]

# ERA5 variables
VARIABLES_WIND = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
VARIABLES_WAVE = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
]

# Time steps (every 6 hours)
TIMES = ["00:00", "06:00", "12:00", "18:00"]
DAYS  = [f"{d:02d}" for d in range(1, 32)]

# ════════════════════════════════════════════════════════════════════
#  REGIONS
#  CDS AREA format: [North, West, South, East]
# ════════════════════════════════════════════════════════════════════

REGIONS = {
    1: {
        "name":        "Houston",
        "area":        [30.0, -95.5, 28.5, -94.0],   # area of the ERA5 files used in the paper
        "era5_dir":    os.path.join(ROOT, "era5"),
        "skip":        False,
        "description": "Houston Ship Channel (main study area)",
    },
    2: {
        "name":        "Chesapeake",
        "area":        [39.6, -77.1, 36.7, -75.4],
        "era5_dir":    os.path.join(ROOT, "era5_chesapeake"),
        "skip":        False,
        "description": "Chesapeake Bay — Baltimore to Norfolk",
        "ais_dir":     os.path.join(ROOT, "ais_chesapeake"),
        "merged_dir":  os.path.join(ROOT, "ais_merged_chesapeake"),
    },
    3: {
        "name":        "NewYork",
        "area":        [41.4, -74.4, 40.3, -73.6],
        "era5_dir":    os.path.join(ROOT, "era5_newyork"),
        "skip":        False,
        "description": "New York Harbor — high traffic density",
        "ais_dir":     os.path.join(ROOT, "ais_newyork"),
        "merged_dir":  os.path.join(ROOT, "ais_merged_newyork"),
    },
    4: {
        "name":        "Mississippi",
        "area":        [30.3, -90.6, 28.7, -88.4],
        "era5_dir":    os.path.join(ROOT, "era5_mississippi"),
        "skip":        False,
        "description": "Mississippi River Delta — New Orleans to Gulf",
        "ais_dir":     os.path.join(ROOT, "ais_mississippi"),
        "merged_dir":  os.path.join(ROOT, "ais_merged_mississippi"),
    },
    5: {
        "name":        "PugetSound",
        "area":        [48.6, -123.1, 46.9, -121.9],
        "era5_dir":    os.path.join(ROOT, "era5_pugetsound"),
        "skip":        False,
        "description": "Puget Sound — Seattle to Tacoma (fjord navigation)",
        "ais_dir":     os.path.join(ROOT, "ais_pugetsound"),
        "merged_dir":  os.path.join(ROOT, "ais_merged_pugetsound"),
    },
}


# ════════════════════════════════════════════════════════════════════
#  ERA5 DOWNLOAD
# ════════════════════════════════════════════════════════════════════

def download_region_era5(region_id: int, c=None):
    """Downloads wind + waves for one region."""
    import cdsapi
    region = REGIONS[region_id]

    if region.get("skip"):
        print(f"  ⏭  Region {region_id} ({region['name']}): skipped")
        return

    era5_dir = Path(region["era5_dir"])
    era5_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ERA5 region {region_id}: {region['name']}")
    print(f"  {region['description']}")
    print(f"  AREA (N/W/S/E): {region['area']}")
    print(f"  Output: {era5_dir}")
    print(f"{'='*60}")

    if c is None:
        c = cdsapi.Client(verify=False)

    for month in MONTHS:
        # ── Wind ──
        wind_file = era5_dir / f"era5_wind_2024_{month}.nc"
        if wind_file.exists() and wind_file.stat().st_size > 500_000:
            print(f"  ⏭  M{month} wind: already exists ({wind_file.stat().st_size // 1024 // 1024} MB)")
        else:
            print(f"  ⬇  M{month} wind...", end=" ", flush=True)
            t0 = time.time()
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": ["reanalysis"],
                        "variable":     VARIABLES_WIND,
                        "year":         ["2024"],
                        "month":        [month],
                        "day":          DAYS,
                        "time":         TIMES,
                        "area":         region["area"],
                        "data_format":  "netcdf",
                        "download_format": "unarchived",
                    },
                    str(wind_file),
                )
                size_mb = wind_file.stat().st_size // 1024 // 1024
                print(f"✅ {size_mb} MB [{time.time()-t0:.0f}s]")
            except Exception as e:
                print(f"❌ {e}")

        # ── Waves ──
        wave_file = era5_dir / f"era5_wave_2024_{month}.nc"
        if wave_file.exists() and wave_file.stat().st_size > 500_000:
            print(f"  ⏭  M{month} waves: already exists ({wave_file.stat().st_size // 1024 // 1024} MB)")
        else:
            print(f"  ⬇  M{month} waves...", end=" ", flush=True)
            t0 = time.time()
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": ["reanalysis"],
                        "variable":     VARIABLES_WAVE,
                        "year":         ["2024"],
                        "month":        [month],
                        "day":          DAYS,
                        "time":         TIMES,
                        "area":         region["area"],
                        "data_format":  "netcdf",
                        "download_format": "unarchived",
                    },
                    str(wave_file),
                )
                size_mb = wave_file.stat().st_size // 1024 // 1024
                print(f"✅ {size_mb} MB [{time.time()-t0:.0f}s]")
            except Exception as e:
                print(f"❌ {e}")

    # Combine the monthly files into one file for the merge scripts
    merge_monthly_nc(era5_dir, region["name"])


def merge_monthly_nc(era5_dir: Path, region_name: str):
    """
    Combines the monthly .nc files into era5_wind_2024.nc and era5_wave_2024.nc,
    the format expected by merge_ais_era5.py and merge_regions.py.
    """
    try:
        import xarray as xr
    except ImportError:
        print("  ⚠️  xarray is not installed, skipping the combination")
        return

    for kind in ["wind", "wave"]:
        monthly = sorted(era5_dir.glob(f"era5_{kind}_2024_*.nc"))
        if not monthly:
            continue
        out = era5_dir / f"era5_{kind}_2024.nc"
        if out.exists():
            print(f"  ⏭  {out.name}: already combined")
            continue
        print(f"  🔗 Combining {len(monthly)} files → {out.name}...", end=" ")
        try:
            ds = xr.open_mfdataset(monthly, combine="by_coords")
            ds.to_netcdf(out)
            ds.close()
            print(f"✅ {out.stat().st_size // 1024 // 1024} MB")
        except Exception as e:
            print(f"❌ {e}")


# ════════════════════════════════════════════════════════════════════
#  AUTO-MERGE AIS + ERA5 for a new region (optional, --merge)
# ════════════════════════════════════════════════════════════════════

def run_merge_for_region(region_id: int):
    """
    Runs merge_ais_era5.py for a new region.
    Called after the ERA5 download when --merge is given.
    """
    import subprocess
    region = REGIONS[region_id]
    if region.get("skip"):
        return

    era5_dir   = Path(region["era5_dir"])
    ais_dir    = Path(region.get("ais_dir", ""))
    merged_dir = Path(region.get("merged_dir", ""))

    wind_nc = era5_dir / "era5_wind_2024.nc"
    wave_nc = era5_dir / "era5_wave_2024.nc"

    if not ais_dir.exists() or not list(ais_dir.glob("*.parquet")):
        print(f"  ⚠️  AIS data not found in {ais_dir}")
        print(f"     Run first: python download_ais_regions.py --region {region_id}")
        return

    if not wind_nc.exists() or not wave_nc.exists():
        print(f"  ⚠️  ERA5 files are not ready in {era5_dir}")
        return

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  🔀 Running merge AIS+ERA5 for {region['name']}...")

    cmd = [
        "python", "merge_ais_era5.py",
        "--ais-dir",   str(ais_dir),
        "--era5-wind", str(wind_nc),
        "--era5-wave", str(wave_nc),
        "--out-dir",   str(merged_dir),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"  ✅ Merge finished → {merged_dir}")
    else:
        print(f"  ❌ Merge finished with an error")


# ════════════════════════════════════════════════════════════════════
#  STATUS
# ════════════════════════════════════════════════════════════════════

def check_status():
    print("\n=== ERA5 DATA STATUS ===\n")
    total_gb = 0.0
    for rid, region in REGIONS.items():
        era5_dir = Path(region["era5_dir"])
        nc_files = list(era5_dir.glob("*.nc")) if era5_dir.exists() else []
        size_mb  = sum(f.stat().st_size for f in nc_files) // 1024 // 1024

        wind_ok  = (era5_dir / "era5_wind_2024.nc").exists()
        wave_ok  = (era5_dir / "era5_wave_2024.nc").exists()
        ready    = wind_ok and wave_ok

        status = "✅ READY" if ready else (
                 "⬇  PARTIAL" if nc_files else "❌ NONE")
        print(f"  [{rid}] {region['name']:15s} {status:15s} {size_mb:5d} MB | {len(nc_files)} .nc files")

        if "merged_dir" in region:
            merged = Path(region["merged_dir"])
            merged_count = len(list(merged.glob("*_era5.parquet"))) if merged.exists() else 0
            print(f"       merged parquet: {merged_count} files")

        total_gb += size_mb / 1024
    print(f"\n  Total ERA5: {total_gb:.1f} GB")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download ERA5 for multiple regions")
    parser.add_argument("--region",  type=int, help="Only one region (1-5)")
    parser.add_argument("--check",   action="store_true", help="File status")
    parser.add_argument("--merge",   action="store_true",
                        help="Run merge AIS+ERA5 after the download")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    try:
        import cdsapi
    except ImportError:
        print("cdsapi is not installed. Installing...")
        os.system("pip install cdsapi")
        import cdsapi

    # One client for all requests
    c = cdsapi.Client(verify=False)

    t_start = time.time()

    if args.region:
        if args.region not in REGIONS:
            print(f"Region {args.region} not found (1-5)")
            return
        download_region_era5(args.region, c)
        if args.merge:
            run_merge_for_region(args.region)
    else:
        print("Starting the ERA5 download for all regions...")
        regions_to_download = [rid for rid, r in REGIONS.items() if not r.get("skip")]
        print(f"Regions: {[REGIONS[r]['name'] for r in regions_to_download]}")
        print("Expected size: ~8-15 GB, time: 3-6 hours\n")

        for rid in regions_to_download:
            download_region_era5(rid, c)
            if args.merge:
                run_merge_for_region(rid)

    elapsed = (time.time() - t_start) / 3600
    print(f"\n✅ ERA5 download finished in {elapsed:.1f} h")
    check_status()


if __name__ == "__main__":
    main()
