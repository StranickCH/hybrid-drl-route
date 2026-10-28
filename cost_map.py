"""
cost_map.py  —  Module 2 of the paper
---------------------------------
Dynamic Cost Map for maritime route optimization.

Cost function:
    C(i,j,t) = α·f_fuel(v, wind) + β·f_risk(wave_hs) + γ·f_time(v)

Where:
    f_fuel  — fuel consumption including wind resistance
    f_risk  — storm risk from the wave height
    f_time  — time cost (inversely proportional to the speed)

Output:
    results/figures/fig_cost_map.png      — cost map
    results/figures/fig_cost_components.png — the three components separately
    results/cost_map_stats.csv            — grid statistics

Usage:
    python cost_map.py
    python cost_map.py --check
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from scipy.ndimage import gaussian_filter


# ─────────────────────────── configuration ───────────────────────────

# Houston Ship Channel bbox
LAT_MIN, LAT_MAX = 29.0, 30.2
LON_MIN, LON_MAX = -95.5, -94.0

# Grid resolution
GRID_ROWS = 60
GRID_COLS = 80

# Cost function weights (α, β, γ) — as in the paper
ALPHA = 0.4   # fuel
BETA  = 0.4   # risk
GAMMA = 0.2   # time

# Physical parameters
V_DESIGN   = 12.0   # knots — design speed of the vessel
RHO_AIR    = 1.225  # kg/m³
C_WIND     = 0.05   # wind resistance coefficient
WAVE_LIMIT = 3.5    # m — safety limit

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
AIS_MERGED  = os.path.join(ROOT, "ais_merged")


# ═══════════════════════════════════════════════════════════════════
#  1. Houston Ship Channel grid
# ═══════════════════════════════════════════════════════════════════

class GridMap:
    """Rectangular grid over the Houston Ship Channel."""

    def __init__(self, lat_min=LAT_MIN, lat_max=LAT_MAX,
                 lon_min=LON_MIN, lon_max=LON_MAX,
                 rows=GRID_ROWS, cols=GRID_COLS):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.rows = rows
        self.cols = cols

        self.lats = np.linspace(lat_min, lat_max, rows)
        self.lons = np.linspace(lon_min, lon_max, cols)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

    def latlon_to_ij(self, lat, lon):
        """Coordinates → cell index."""
        i = int((lat - self.lat_min) / (self.lat_max - self.lat_min) * (self.rows - 1))
        j = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * (self.cols - 1))
        return np.clip(i, 0, self.rows-1), np.clip(j, 0, self.cols-1)

    def ij_to_latlon(self, i, j):
        return self.lats[i], self.lons[j]


# ═══════════════════════════════════════════════════════════════════
#  2. Aggregation of the AIS data on the grid
# ═══════════════════════════════════════════════════════════════════

def aggregate_to_grid(grid: GridMap, n_files: int = 50) -> dict:
    """
    Reads the AIS+ERA5 files and aggregates the meteorological parameters on the grid.
    Returns a dict of numpy arrays (rows × cols).
    """
    files = sorted(Path(AIS_MERGED).glob("*_era5.parquet"))[:n_files]
    print(f"Aggregating {len(files)} files on the {grid.rows}×{grid.cols} grid...")

    # Accumulators
    sum_wind_u  = np.zeros((grid.rows, grid.cols))
    sum_wind_v  = np.zeros((grid.rows, grid.cols))
    sum_wind_sp = np.zeros((grid.rows, grid.cols))
    sum_wave_hs = np.zeros((grid.rows, grid.cols))
    sum_sog     = np.zeros((grid.rows, grid.cols))
    count       = np.zeros((grid.rows, grid.cols))   # all reports in the cell
    counts      = {k: np.zeros((grid.rows, grid.cols)) for k in ("wind_u", "wind_v", "wind_speed", "wave_hs", "SOG")}

    for f in files:
        df = pd.read_parquet(f)

        lat_col = next((c for c in ["LAT","lat","latitude"] if c in df.columns), None)
        lon_col = next((c for c in ["LON","lon","longitude"] if c in df.columns), None)
        if not lat_col or not lon_col:
            continue

        df = df.dropna(subset=[lat_col, lon_col])
        df = df[
            df[lat_col].between(grid.lat_min, grid.lat_max) &
            df[lon_col].between(grid.lon_min, grid.lon_max)
        ]
        if df.empty:
            continue

        # Cell indices
        i_arr = ((df[lat_col] - grid.lat_min) /
                 (grid.lat_max - grid.lat_min) * (grid.rows - 1)).astype(int).clip(0, grid.rows-1)
        j_arr = ((df[lon_col] - grid.lon_min) /
                 (grid.lon_max - grid.lon_min) * (grid.cols - 1)).astype(int).clip(0, grid.cols-1)

        for col_name, arr in [
            ("wind_u",     sum_wind_u),
            ("wind_v",     sum_wind_v),
            ("wind_speed", sum_wind_sp),
            ("wave_hs",    sum_wave_hs),
            ("SOG",        sum_sog),
        ]:
            if col_name in df.columns:
                vals = df[col_name].values.astype(float)
                ok = ~np.isnan(vals)   # missing values are not counted (they used to be counted as 0)
                np.add.at(arr, (i_arr.values[ok], j_arr.values[ok]), vals[ok])
                np.add.at(counts[col_name], (i_arr.values[ok], j_arr.values[ok]), 1)

        np.add.at(count, (i_arr.values, j_arr.values), 1)

    # Mean
    mask = count > 0
    result = {}
    for name, arr, cnt in [
        ("wind_u",     sum_wind_u,  counts["wind_u"]),
        ("wind_v",     sum_wind_v,  counts["wind_v"]),
        ("wind_speed", sum_wind_sp, counts["wind_speed"]),
        ("wave_hs",    sum_wave_hs, counts["wave_hs"]),
        ("sog",        sum_sog,     counts["SOG"]),
        ("count",      count,       count),
    ]:
        avg = np.zeros_like(arr)
        m = cnt > 0
        avg[m] = arr[m] / cnt[m]
        # Gaussian smoothing — removes sparsity artefacts
        avg = gaussian_filter(avg, sigma=1.5)
        result[name] = avg

    print(f"  Cells with data: {mask.sum():,} / {grid.rows * grid.cols}")
    return result


# ═══════════════════════════════════════════════════════════════════
#  3. Cost components
# ═══════════════════════════════════════════════════════════════════

def fuel_cost(wind_speed: np.ndarray, wind_u: np.ndarray,
              wind_v: np.ndarray, v_ship: float = V_DESIGN) -> np.ndarray:
    """
    f_fuel = base consumption + wind resistance.
    Normalised to [0, 1].
    """
    # Angle between the ship course (assumed N→S, 180°) and the wind
    ship_dir = np.array([0.0, -1.0])  # south
    wind_mag = np.maximum(wind_speed, 0.01)

    # Projection of the wind on the direction of motion (head wind = worse)
    head_wind = -(wind_u * ship_dir[0] + wind_v * ship_dir[1]) / wind_mag

    # Additional resistance force proportional to wind²
    extra = C_WIND * RHO_AIR * wind_speed**2 * np.clip(head_wind, 0, 1)

    # Base consumption + wind part
    base = v_ship**3  # proportional to the cube of the speed (Admiralty formula)
    cost = base + extra * 1e-2

    return _normalize(cost)


def risk_cost(wave_hs: np.ndarray, limit: float = WAVE_LIMIT) -> np.ndarray:
    """
    f_risk = sigmoid of the wave height.
    For wave_hs > limit → risk → 1.
    """
    # Sigmoid centred at WAVE_LIMIT/2
    k = 4.0 / limit
    cost = 1.0 / (1.0 + np.exp(-k * (wave_hs - limit / 2)))
    return _normalize(cost)


def time_cost(sog: np.ndarray, v_design: float = V_DESIGN) -> np.ndarray:
    """
    f_time = inversely proportional to SOG.
    Slow zones are more expensive.
    """
    sog_safe = np.maximum(sog, 0.5)
    cost = v_design / sog_safe
    return _normalize(cost)


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_cost_map(grid_data: dict,
                     alpha: float = ALPHA,
                     beta:  float = BETA,
                     gamma: float = GAMMA) -> dict:
    """
    Computes the total cost map and the three components.
    """
    f_fuel = fuel_cost(
        grid_data["wind_speed"],
        grid_data["wind_u"],
        grid_data["wind_v"],
    )
    f_risk = risk_cost(grid_data["wave_hs"])
    f_time = time_cost(grid_data["sog"])

    total = alpha * f_fuel + beta * f_risk + gamma * f_time
    total = _normalize(total)

    return {
        "total":  total,
        "f_fuel": f_fuel,
        "f_risk": f_risk,
        "f_time": f_time,
    }


# ═══════════════════════════════════════════════════════════════════
#  4. Plots
# ═══════════════════════════════════════════════════════════════════

# The two ports of the Houston Ship Channel
PORT_HOUSTON  = (29.73, -95.27)   # Port of Houston
PORT_GALVESTON = (29.31, -94.79)  # Galveston

def plot_cost_map(cost_maps: dict, grid: GridMap, figures_dir: str):
    """
    Main figure — total cost map with wind arrows.
    Publication quality, IEEE style.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    total = cost_maps["total"]

    # Colour map
    cmap = plt.cm.RdYlGn_r   # green = cheap, red = expensive
    im = ax.pcolormesh(
        grid.lon_grid, grid.lat_grid, total,
        cmap=cmap, vmin=0, vmax=1, shading="auto", alpha=0.85
    )

    # Cost contours
    cs = ax.contour(
        grid.lon_grid, grid.lat_grid, total,
        levels=[0.3, 0.5, 0.7, 0.85],
        colors="white", linewidths=0.7, alpha=0.6
    )
    ax.clabel(cs, fmt="%.1f", fontsize=7, colors="white")

    # Wind arrows (quiver) — every 6 cells
    step = 6
    ax.quiver(
        grid.lon_grid[::step, ::step],
        grid.lat_grid[::step, ::step],
        cost_maps.get("wind_u", np.zeros_like(total))[::step, ::step],
        cost_maps.get("wind_v", np.zeros_like(total))[::step, ::step],
        alpha=0.5, color="white", scale=40, width=0.003,
        headwidth=3, headlength=4,
    )

    # Ports
    for (lat, lon), name, marker in [
        (PORT_HOUSTON,   "Port of\nHouston",   "^"),
        (PORT_GALVESTON, "Galveston",           "s"),
    ]:
        ax.plot(lon, lat, marker, color="white",
                markersize=10, markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    # Example route (direct, for illustration)
    route_lons = np.linspace(PORT_HOUSTON[1],   PORT_GALVESTON[1], 30)
    route_lats = np.linspace(PORT_HOUSTON[0],   PORT_GALVESTON[0], 30)
    ax.plot(route_lons, route_lats, "w--", linewidth=1.5,
            alpha=0.8, label="Reference route", zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized Cost C(i,j,t)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_title(
        "Dynamic Cost Map — Houston Ship Channel\n"
        f"α={ALPHA} (fuel) + β={BETA} (risk) + γ={GAMMA} (time)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    fig.tight_layout()
    path = Path(figures_dir) / "fig_cost_map.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {path}")


def plot_cost_components(cost_maps: dict, grid_data: dict,
                         grid: GridMap, figures_dir: str):
    """
    4 subplots: f_fuel, f_risk, f_time, total C.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    panels = [
        (cost_maps["f_fuel"], "Fuel Cost  f_fuel",   "YlOrRd",  f"α = {ALPHA}"),
        (cost_maps["f_risk"], "Risk Cost  f_risk",   "RdPu",    f"β = {BETA}"),
        (cost_maps["f_time"], "Time Cost  f_time",   "Blues_r", f"γ = {GAMMA}"),
        (cost_maps["total"],  "Total Cost  C(i,j,t)","RdYlGn_r","α+β+γ weighted"),
    ]

    for ax, (data, title, cmap, subtitle) in zip(axes.flat, panels):
        im = ax.pcolormesh(
            grid.lon_grid, grid.lat_grid, data,
            cmap=cmap, vmin=0, vmax=1, shading="auto"
        )
        ax.contour(grid.lon_grid, grid.lat_grid, data,
                   levels=4, colors="white", linewidths=0.5, alpha=0.4)

        # Ports
        for (lat, lon), marker in [(PORT_HOUSTON, "^"), (PORT_GALVESTON, "s")]:
            ax.plot(lon, lat, marker, color="white", markersize=7,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"{title}\n({subtitle})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle(
        "Cost Map Components — Houston Ship Channel",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    path = Path(figures_dir) / "fig_cost_components.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {path}")


def plot_cost_distribution(cost_maps: dict, figures_dir: str):
    """
    Histograms of every component.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    data_list = [
        (cost_maps["f_fuel"].flatten(), "Fuel cost",  "#e74c3c"),
        (cost_maps["f_risk"].flatten(), "Risk cost",  "#9b59b6"),
        (cost_maps["f_time"].flatten(), "Time cost",  "#3498db"),
    ]

    for ax, (data, label, color) in zip(axes, data_list):
        ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean={data.mean():.2f}")
        ax.set_xlabel("Normalized cost", fontsize=9)
        ax.set_ylabel("Cell count", fontsize=9)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribution of Cost Components", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = Path(figures_dir) / "fig_cost_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  5. Statistics for the paper
# ═══════════════════════════════════════════════════════════════════

def save_stats(cost_maps: dict, results_dir: str):
    rows = []
    for name, arr in cost_maps.items():
        rows.append({
            "component": name,
            "mean":  round(arr.mean(), 4),
            "std":   round(arr.std(),  4),
            "min":   round(arr.min(),  4),
            "max":   round(arr.max(),  4),
            "p25":   round(np.percentile(arr, 25), 4),
            "p75":   round(np.percentile(arr, 75), 4),
        })
    df = pd.DataFrame(rows)
    path = Path(results_dir) / "cost_map_stats.csv"
    df.to_csv(path, index=False)
    print(f"✅ Statistics → {path}")
    print(df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
#  6. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta",  type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    args = parser.parse_args()

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Grid
    grid = GridMap()

    # Data aggregation
    grid_data = aggregate_to_grid(grid, n_files=100)

    if args.check:
        print("\n=== Check ===")
        for k, v in grid_data.items():
            print(f"  {k}: min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f}")
        return

    # Cost map
    cost_maps = compute_cost_map(grid_data, args.alpha, args.beta, args.gamma)
    # Add the wind for the quiver
    cost_maps["wind_u"] = grid_data["wind_u"]
    cost_maps["wind_v"] = grid_data["wind_v"]

    # Plots
    plot_cost_map(cost_maps, grid, FIGURES_DIR)
    plot_cost_components(cost_maps, grid_data, grid, FIGURES_DIR)
    plot_cost_distribution(cost_maps, FIGURES_DIR)

    # Statistics
    save_stats({k: v for k, v in cost_maps.items()
                if k not in ("wind_u", "wind_v")}, RESULTS_DIR)

    print("\n✅ Module 2 done!")
    print(f"   Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
