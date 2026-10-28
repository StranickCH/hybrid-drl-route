# -*- coding: utf-8 -*-
"""Packs the results behind the paper into zip archives for Zenodo.

paper_results.zip  - LSTM model, scalers and metrics; PPO models (best checkpoints) and training
                     logs of all scenarios and ablation variants; raw evaluation results; SAC
                     results; tables and figures.
merged_data.zip    - (with --data) the merged AIS+ERA5 parquet files of all regions and the ERA5
                     files, i.e. everything needed to rerun the models without downloading the
                     ~60 GB of raw AIS data.

Run:  python pack_results.py [--data] [--root FOLDER [FOLDER ...]]
      --root: folders with results/, sac_results/ and the data folders (default: this folder).
      Several folders can be given (e.g. the LSTM run and the RL run); a file present in more
      than one folder is taken from the last one. The archives are written next to this script.
"""
import argparse
import glob
import os
import zipfile

RESULTS = [
    # LSTM (Module 1)
    "results/metrics_lstm_temporal.csv", "results/metrics_baselines_temporal.csv",
    "results/lstm_model_temporal.pt", "results/scaler_x_temporal.pkl", "results/scaler_y_temporal.pkl",
    "results/figures/fig_loss_curve_temporal.png", "results/figures/fig_predictions_temporal.png",
    "results/figures/fig_predictions_v2.png", "results/figures/fig_error_by_horizon.png",
    "results/figures/ieee/fig_mc_dropout.*", "results/figures/ieee/fig_uncertainty.*",
    "results/figures/ieee/fig_generalization.*",
    # cost map, PPO experiments (Modules 2 and 3)
    "results/cost_map_stats.csv",
    "results/experiments/summary_table.csv", "results/experiments/summary_pathcost.csv",
    "results/experiments/*/raw_results.json", "results/experiments/*/ttest_results.csv",
    "results/experiments/*/ppo_seed*.zip", "results/experiments/*/ppo_seed*_monitor.monitor.csv",
    "results/experiments/*/ppo_seed*_best/evaluations.npz",
    # ablation
    "results/ablation/ablation_raw.json", "results/ablation/ablation_pathcost.csv",
    "results/ablation/*/model_seed*.zip", "results/ablation/*/model_seed*_monitor.monitor.csv",
    "results/ablation/*/model_seed*_best/evaluations.npz",
    # SAC
    "sac_results/*.csv", "sac_results/*.png", "sac_results/*.pdf",
    "sac_results/*/seed*.json", "sac_results/*/seed*_actor.pt",
    # figures and tables of the paper
    "results/figures/paper/*",
]

DATA = [
    "ais_merged/*_era5.parquet", "ais_merged_chesapeake/*_era5.parquet", "ais_merged_newyork/*_era5.parquet",
    "ais_merged_mississippi/*_era5.parquet", "ais_merged_pugetsound/*_era5.parquet",
    "era5/era5_wind_2024.nc", "era5/era5_wave_2024.nc", "era5_*/era5_wind_2024.nc", "era5_*/era5_wave_2024.nc",
]


def pack(roots, patterns, out):
    files = {}   # archive path -> file on disk (the last root wins)
    for p in patterns:
        found = False
        for root in roots:
            for f in sorted(glob.glob(os.path.join(root, p))):
                if os.path.isfile(f):
                    files[os.path.relpath(f, root).replace("\\", "/")] = f
                    found = True
        if not found:
            print("not found:", p)
    n_bytes = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, f in sorted(files.items()):
            z.write(f, arc)
            n_bytes += os.path.getsize(f)
    print(f"{len(files)} files, {n_bytes / 1e6:.0f} MB -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", nargs="+", default=[here])
    ap.add_argument("--data", action="store_true", help="also build merged_data.zip")
    args = ap.parse_args()
    pack(args.root, RESULTS, os.path.join(here, "paper_results.zip"))
    if args.data:
        pack(args.root, DATA, os.path.join(here, "merged_data.zip"))


if __name__ == "__main__":
    main()
