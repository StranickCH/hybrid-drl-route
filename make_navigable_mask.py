"""
make_navigable_mask.py
----------------------
Builds navigable_mask.npy (GRID_ROWS x GRID_COLS, bool): the cells a vessel may enter.

A cell is navigable if
  (a) less than half of it is land in the global 1-km land mask of the `global-land-mask`
      package (GLOBE elevation data), or
  (b) it contains AIS positions in the files used for the cost map, or touches such a cell
      (narrow channels such as the Houston Ship Channel are narrower than the 1-km land mask;
      the one-cell buffer lets a vessel with a continuous heading follow a channel that is one
      ~1.2-nm cell wide on the grid);
and it belongs to the 8-connected water component that contains START and GOAL.

Run once (the mask is shipped with the repository):
  pip install global-land-mask
  python make_navigable_mask.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from global_land_mask import globe
from scipy.ndimage import binary_dilation, label

import run_experiments as rx

N_SUB = 7   # sub-samples per cell side for the land fraction


def land_fraction():
    dlat = (rx.LAT_MAX - rx.LAT_MIN) / (rx.GRID_ROWS - 1)   # cell (i, j) as defined by latlon_to_ij
    dlon = (rx.LON_MAX - rx.LON_MIN) / (rx.GRID_COLS - 1)
    k = (np.arange(N_SUB) + 0.5) / N_SUB
    frac = np.zeros((rx.GRID_ROWS, rx.GRID_COLS))
    for i in range(rx.GRID_ROWS):
        for j in range(rx.GRID_COLS):
            la, lo = np.meshgrid(rx.LAT_MIN + (i + k) * dlat, rx.LON_MIN + (j + k) * dlon, indexing="ij")
            frac[i, j] = globe.is_land(la, lo).mean()
    return frac


def ais_count():
    """AIS positions per cell in the same files as build_cost_map."""
    cnt = np.zeros((rx.GRID_ROWS, rx.GRID_COLS))
    for f in sorted(Path(rx.AIS_MERGED).glob("*_era5.parquet"))[:80]:
        df = pd.read_parquet(f, columns=["LAT", "LON", "SOG"]).dropna()
        df = df[df["LAT"].between(rx.LAT_MIN, rx.LAT_MAX) & df["LON"].between(rx.LON_MIN, rx.LON_MAX)]
        i = ((df["LAT"] - rx.LAT_MIN) / (rx.LAT_MAX - rx.LAT_MIN) * (rx.GRID_ROWS - 1)).astype(int).clip(0, rx.GRID_ROWS - 1)
        j = ((df["LON"] - rx.LON_MIN) / (rx.LON_MAX - rx.LON_MIN) * (rx.GRID_COLS - 1)).astype(int).clip(0, rx.GRID_COLS - 1)
        np.add.at(cnt, (i.values, j.values), 1)
    return cnt


def main():
    water = (land_fraction() <= 0.5) | binary_dilation(ais_count() > 0, structure=np.ones((3, 3)))
    lab, _ = label(water, structure=np.ones((3, 3)))
    s, g = rx.latlon_to_ij(*rx.START), rx.latlon_to_ij(*rx.GOAL)
    assert lab[s] > 0 and lab[s] == lab[g], "START and GOAL are not connected by water"
    nav = lab == lab[s]
    np.save(os.path.join(rx.ROOT, "navigable_mask.npy"), nav)
    print(f"navigable cells: {nav.sum()} of {nav.size} ({nav.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
