"""
download_ais_regions.py  (v2)
--------------------------------------------------
Two ways to download the 2024 AIS data for the additional regions:

METHOD 1 (automatic): tries the URL templates below (Marine Cadastre Azure Blob, NOAA daily/zone ZIP)
  URL: https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/aistrack/

METHOD 2 (fallback): AccessAIS web tool - "clip and ship" by bbox
  https://marinecadastre.gov/accessais/
  If the automatic download fails, the script prints instructions for a manual download.

Usage:
  python download_ais_regions.py           # all regions
  python download_ais_regions.py --region 2
  python download_ais_regions.py --check
  python download_ais_regions.py --probe   # find a working URL template
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
import zipfile
import json
from pathlib import Path

import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

AIS_RAW_DIR = os.path.join(ROOT, "ais_raw_regions")
MONTHS      = [1, 2, 3, 4, 5, 6]
YEAR        = 2024

# Candidate URL templates (in order of priority)
URL_TEMPLATES = [
    # New 2024/2025 format - Azure Blob, monthly GeoParquet
    "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/aistrack/AIS_{year}_{month:02d}.parquet",
    # Old format - NOAA daily ZIP (first day of each month)
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_01.zip",
    # Zone format (if the daily file does not work)
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_Zone15.zip",
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_Zone18.zip",
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_Zone10.zip",
]

# AccessAIS API (clip and ship)
ACCESS_AIS_API = "https://www.marinecadastre.gov/accessais/api/data"
ACCESS_AIS_WEB = "https://marinecadastre.gov/accessais/"

REGIONS = {
    2: {
        "name":        "Chesapeake_Bay",
        "display":     "Chesapeake Bay",
        "lat_min":     36.8,  "lat_max": 39.5,
        "lon_min":    -77.0,  "lon_max": -75.5,
        "out_dir":     os.path.join(ROOT, "ais_chesapeake"),
        "description": "Baltimore → Norfolk",
    },
    3: {
        "name":        "New_York_Harbor",
        "display":     "New York Harbor",
        "lat_min":     40.4,  "lat_max": 41.3,
        "lon_min":    -74.3,  "lon_max": -73.7,
        "out_dir":     os.path.join(ROOT, "ais_newyork"),
        "description": "NY/NJ Harbor → Long Island Sound",
    },
    4: {
        "name":        "Mississippi_Delta",
        "display":     "Mississippi Delta",
        "lat_min":     28.8,  "lat_max": 30.2,
        "lon_min":    -90.5,  "lon_max": -88.5,
        "out_dir":     os.path.join(ROOT, "ais_mississippi"),
        "description": "New Orleans → Gulf of Mexico",
    },
    5: {
        "name":        "Puget_Sound",
        "display":     "Puget Sound",
        "lat_min":     47.0,  "lat_max": 48.5,
        "lon_min":   -123.0,  "lon_max": -122.0,
        "out_dir":     os.path.join(ROOT, "ais_pugetsound"),
        "description": "Seattle → Tacoma",
    },
}


# ════════════════════════════════════════════════════════════════════
#  METHOD 1: automatic download
# ════════════════════════════════════════════════════════════════════

def probe_working_url(year=2024, month=1):
    """Checks which URL template works and returns it."""
    print("Checking available URL templates...")
    for template in URL_TEMPLATES:
        url = template.format(year=year, month=month)
        fname = url.split("/")[-1]
        try:
            req  = urllib.request.Request(url, method="HEAD")
            resp = urllib.request.urlopen(req, timeout=15)
            if resp.status == 200:
                size = resp.headers.get("Content-Length", "?")
                print(f"  ✅ Works: {fname}  ({int(size)//1024//1024 if size != '?' else '?'} MB)")
                return template
        except urllib.error.HTTPError as e:
            print(f"  ❌ {fname}: HTTP {e.code}")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")
    return None


def download_file(url, dest, label=""):
    """Downloads a file with a progress indicator."""
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  ⏭  {label}: already exists ({dest.stat().st_size // 1024 // 1024} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size * 100, 100)
                mb  = count * block_size / 1024 / 1024
                print(f"\r  ⬇  {label}: {pct:.0f}% ({mb:.0f} MB)", end="", flush=True)
            else:
                mb = count * block_size / 1024 / 1024
                print(f"\r  ⬇  {label}: {mb:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, str(tmp), reporthook)
        tmp.rename(dest)
        size_mb = dest.stat().st_size // 1024 // 1024
        print(f"\r  ✅ {label}: {size_mb} MB → {dest.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"\r  ⚠️  {label}: HTTP {e.code}")
        if tmp.exists(): tmp.unlink()
        return False
    except Exception as e:
        print(f"\r  ❌ {label}: {e}")
        if tmp.exists(): tmp.unlink()
        return False


def filter_parquet_by_bbox(src_path, region, out_dir):
    """Filters a GeoParquet/Parquet file by the region bbox."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (src_path.stem + f"_{region['name']}.parquet")

    if out_path.exists():
        return 0

    try:
        df = pd.read_parquet(src_path)
        lat_col = next((c for c in ["LAT","lat","latitude","LATITUDE"] if c in df.columns), None)
        lon_col = next((c for c in ["LON","lon","longitude","LONGITUDE"] if c in df.columns), None)

        if not lat_col or not lon_col:
            # Try the geometry column (GeoParquet)
            if "geometry" in df.columns:
                try:
                    import geopandas as gpd
                    gdf = gpd.read_parquet(src_path)
                    gdf = gdf.cx[region["lon_min"]:region["lon_max"],
                                 region["lat_min"]:region["lat_max"]]
                    if not gdf.empty:
                        gdf.to_parquet(out_path, index=False)
                        return len(gdf)
                except ImportError:
                    print("    geopandas is not installed, skipping GeoParquet")
            return 0

        df = df.rename(columns={lat_col: "LAT", lon_col: "LON"})
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        mask = (df["LAT"].between(region["lat_min"], region["lat_max"]) &
                df["LON"].between(region["lon_min"], region["lon_max"]))
        df = df[mask].dropna(subset=["LAT","LON"])

        if not df.empty:
            df.to_parquet(out_path, index=False)
            return len(df)
    except Exception as e:
        print(f"    ❌ Filtering error: {e}")
    return 0


def extract_and_filter_zip(zip_path, region, out_dir):
    """Unpacks a ZIP with CSV and filters it by bbox."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (zip_path.stem + f"_{region['name']}.parquet")

    if out_path.exists():
        return 0

    total = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            dfs = []
            for csv_name in csv_files:
                with zf.open(csv_name) as f:
                    try:
                        df = pd.read_csv(f, low_memory=False)
                    except Exception:
                        continue

                df.columns = [c.strip() for c in df.columns]
                lat_col = next((c for c in ["LAT","lat","latitude"] if c in df.columns), None)
                lon_col = next((c for c in ["LON","lon","longitude"] if c in df.columns), None)
                if not lat_col or not lon_col:
                    continue

                df = df.rename(columns={lat_col:"LAT", lon_col:"LON"})
                df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
                df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
                mask = (df["LAT"].between(region["lat_min"], region["lat_max"]) &
                        df["LON"].between(region["lon_min"], region["lon_max"]))
                df = df[mask]
                if not df.empty:
                    dfs.append(df)

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                combined.to_parquet(out_path, index=False)
                total = len(combined)

    except zipfile.BadZipFile:
        print(f"    ❌ Corrupted ZIP: {zip_path.name}")

    return total


# ════════════════════════════════════════════════════════════════════
#  METHOD 2: AccessAIS API (clip and ship)
# ════════════════════════════════════════════════════════════════════

def try_access_ais_api(region, out_dir, month):
    """
    Tries the AccessAIS REST API to download data by bbox.
    If the API does not work, prints a link for a manual download.
    """
    import urllib.parse

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Request parameters
    params = {
        "minLon": region["lon_min"],
        "maxLon": region["lon_max"],
        "minLat": region["lat_min"],
        "maxLat": region["lat_max"],
        "startDate": f"2024-{month:02d}-01",
        "endDate":   f"2024-{month:02d}-28",
        "dataType":  "points",
    }

    url = ACCESS_AIS_API + "?" + urllib.parse.urlencode(params)
    out_path = out_dir / f"AIS_{YEAR}_{month:02d}_{region['name']}_api.parquet"

    if out_path.exists():
        return True

    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())

        if data and "features" in data:
            rows = []
            for feat in data["features"]:
                props = feat.get("properties", {})
                geom  = feat.get("geometry", {})
                if geom and geom.get("coordinates"):
                    props["LON"] = geom["coordinates"][0]
                    props["LAT"] = geom["coordinates"][1]
                rows.append(props)
            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(out_path, index=False)
                print(f"  ✅ AccessAIS API: {len(df):,} rows → M{month:02d}")
                return True
    except Exception:
        pass

    return False


def print_manual_download_instructions(region):
    """Prints instructions for a manual download via AccessAIS."""
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  MANUAL DOWNLOAD: {region['display']:35s} ║
  ╠══════════════════════════════════════════════════════════╣
  ║  1. Open: https://marinecadastre.gov/accessais/         ║
  ║  2. Draw a box or enter the coordinates:                ║
  ║     North: {region['lat_max']:<10}  South: {region['lat_min']:<10}          ║
  ║     West:  {region['lon_min']:<10}  East:  {region['lon_max']:<10}          ║
  ║  3. Period: Jan 2024 – Jun 2024                         ║
  ║  4. Click Download → CSV                                ║
  ║  5. Save to: {region['out_dir'][:40]:40s}   ║
  ╚══════════════════════════════════════════════════════════╝
""")


# ════════════════════════════════════════════════════════════════════
#  MAIN DOWNLOAD FUNCTION
# ════════════════════════════════════════════════════════════════════

def download_region(region_id, url_template=None):
    region = REGIONS[region_id]
    print(f"\n{'='*58}")
    print(f"  Region {region_id}: {region['display']} — {region['description']}")
    print(f"{'='*58}")

    raw_dir = Path(AIS_RAW_DIR) / region["name"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(region["out_dir"])

    total_rows  = 0
    any_success = False

    for month in MONTHS:
        label = f"M{month:02d}"

        # ── Attempt 1: download the file from the URL ──
        if url_template:
            url   = url_template.format(year=YEAR, month=month)
            fname = url.split("/")[-1]
            dest  = raw_dir / fname

            ok = download_file(url, dest, label)
            if ok and dest.exists():
                any_success = True
                # Conversion depending on the format
                if dest.suffix == ".parquet":
                    rows = filter_parquet_by_bbox(dest, region, out_dir)
                else:
                    rows = extract_and_filter_zip(dest, region, out_dir)
                total_rows += rows
                if rows > 0:
                    print(f"    → {rows:,} rows in bbox {region['display']}")
                continue

        # ── Attempt 2: AccessAIS API ──
        api_ok = try_access_ais_api(region, out_dir, month)
        if api_ok:
            any_success = True
            continue

    # ── If nothing worked - instructions for a manual download ──
    parquets = list(out_dir.glob("*.parquet")) if out_dir.exists() else []
    if not parquets:
        print_manual_download_instructions(region)
    else:
        print(f"\n  ✅ {region['display']}: {len(parquets)} files, ~{total_rows:,} rows")
        print(f"     → {out_dir}")


# ════════════════════════════════════════════════════════════════════
#  STATUS
# ════════════════════════════════════════════════════════════════════

def check_status():
    print("\n=== AIS DATA STATUS ===\n")
    for rid, region in REGIONS.items():
        out_dir = Path(region["out_dir"])
        pq   = list(out_dir.glob("*.parquet")) if out_dir.exists() else []
        size = sum(f.stat().st_size for f in pq) // 1024 // 1024 if pq else 0
        ok   = len(pq) >= 3
        print(f"  {'✅' if ok else '❌'} [{rid}] {region['display']:20s} "
              f"{len(pq):3d} files  {size:5d} MB  → {region['out_dir']}")
    print()


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=int, help="Only one region (2-5)")
    parser.add_argument("--check",  action="store_true", help="Data status")
    parser.add_argument("--probe",  action="store_true", help="Find a working URL")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    # Find a working URL template
    url_template = probe_working_url()

    if args.probe:
        if url_template:
            print(f"\nWorking template: {url_template}")
        else:
            print("\n⚠️  No URL works.")
            print("   Use AccessAIS manually: https://marinecadastre.gov/accessais/")
        return

    if not url_template:
        print("\n⚠️  Automatic download is not available.")
        print("   Printing instructions for a manual download:\n")
        for rid, region in REGIONS.items():
            print_manual_download_instructions(region)
        return

    if args.region:
        if args.region not in REGIONS:
            print(f"Region {args.region} not found. Available: {list(REGIONS.keys())}")
            return
        download_region(args.region, url_template)
    else:
        print(f"\nDownloading AIS for {len(REGIONS)} regions...\n")
        for rid in REGIONS:
            download_region(rid, url_template)

    check_status()


if __name__ == "__main__":
    main()
