"""
ablation_study.py  (environment v2)
-----------------------------------
Ablation study: what the cost map and the weather in the observation contribute to the PPO agent.

All variants use the environment, reward and PPO settings of run_experiments.py; they differ only
in the map the agent is trained on and in the weather/traffic entries of its observation:
  V1: No Map            — uniform cost map (0.5 in every water cell, land as in the full map):
                          the agent can only minimise distance
  V2: Traffic Map       — map from the mean SOG of the first 20 AIS files only, no ERA5
  V3: Full Map          — full AIS+ERA5 cost map (Normal scenario), no weather in the observation
  V4: Full Map + State  — full map and the local wind, wave and traffic speed in the observation
                          (the full system of the paper)
Every trained agent is evaluated on the full AIS+ERA5 map with the path-integrated cost
(cost·nm, path_cost_nm in run_experiments.py), so that all variants are compared on one scale.
The unit of independence is the training seed.

Usage:
  python ablation_study.py

Output:
  results/ablation/ablation_raw.json, ablation_pathcost.csv, V*/model_seed*.zip
  results/figures/ieee/fig_ablation.png + .pdf, ablation_table.csv
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

import run_experiments as rx

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here
AIS_MERGED = os.path.join(ROOT, "ais_merged")
ABL_DIR    = os.path.join(ROOT, "results", "ablation")
FIG_DIR    = os.path.join(ROOT, "results", "figures", "ieee")

N_SEEDS    = 10
PPO_STEPS  = rx.PPO_TIMESTEPS
N_EVAL_EPS = rx.N_EVAL_EPISODES

COLORS = {"V1: No Map": "#e74c3c", "V2: Traffic Map": "#f39c12",
          "V3: Full Map": "#3498db", "V4: Full Map + State": "#27ae60"}


def build_traffic_cost_map():
    """Map from the mean SOG of the first 20 AIS files only (no ERA5): slow cells are expensive."""
    files = sorted(Path(AIS_MERGED).glob("*_era5.parquet"))[:20]
    sum_sog = np.zeros((rx.GRID_ROWS, rx.GRID_COLS))
    count   = np.zeros((rx.GRID_ROWS, rx.GRID_COLS))
    for f in files:
        df = pd.read_parquet(f, columns=["LAT", "LON", "SOG"])
        df = df[df["LAT"].between(rx.LAT_MIN, rx.LAT_MAX) & df["LON"].between(rx.LON_MIN, rx.LON_MAX)].dropna()
        i = ((df["LAT"] - rx.LAT_MIN) / (rx.LAT_MAX - rx.LAT_MIN) * (rx.GRID_ROWS - 1)).astype(int).clip(0, rx.GRID_ROWS - 1)
        j = ((df["LON"] - rx.LON_MIN) / (rx.LON_MAX - rx.LON_MIN) * (rx.GRID_COLS - 1)).astype(int).clip(0, rx.GRID_COLS - 1)
        np.add.at(sum_sog, (i.values, j.values), df["SOG"].values)
        np.add.at(count, (i.values, j.values), 1)
    m = count > 0
    sog = np.zeros_like(sum_sog)
    sog[m] = sum_sog[m] / count[m]
    sog = gaussian_filter(sog, sigma=1.5)
    c = 12.0 / np.maximum(sog, 0.5)
    c = ((c - c.min()) / (c.max() - c.min())).astype(np.float32)
    if rx.NAV is not None:
        c[~rx.NAV] = 1.0   # land, as in the full map
    return c


VARIANTS = ["V1: No Map", "V2: Traffic Map", "V3: Full Map", "V4: Full Map + State"]


def variant_dir(v):
    return v.replace(" ", "_").replace(":", "").replace("+", "plus")


def variant_maps():
    """{variant: (training map, fields or None)}"""
    full_map, full_fields = rx.build_cost_map(**rx.SCENARIOS["Normal"], return_fields=True)
    return {
        "V1: No Map":           (np.full_like(full_map, 0.5) if rx.NAV is None
                                 else np.where(rx.NAV, 0.5, 1.0).astype(np.float32), None),   # land only
        "V2: Traffic Map":      (build_traffic_cost_map(), None),
        "V3: Full Map":         (full_map, None),
        "V4: Full Map + State": (full_map, full_fields),
    }


def train_and_eval(variant, train_map, fields, seed, save_dir, eval_map):
    """Trains (or loads) PPO on train_map; evaluates the trajectories on eval_map (path-integrated cost)."""
    from stable_baselines3 import PPO
    model_path = save_dir / f"model_seed{seed}.zip"
    if model_path.exists():
        model = PPO.load(str(model_path), env=rx.MaritimeEnv(train_map, fields=fields), device="cpu")
        how = "loaded"
    else:
        model = rx.train_ppo_seed(train_map, seed, PPO_STEPS, model_path, fields=fields)
        how = "trained"
    env = rx.MaritimeEnv(train_map, seed=seed * 1000, fields=fields)
    costs, nms, arrived = [], [], []
    for _ in range(N_EVAL_EPS):
        obs, _ = env.reset()
        done, info = False, {"arrived": False}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        c, nm = rx.path_cost_nm(list(env.trajectory) + [rx.GOAL], eval_map)   # route completed to the goal
        costs.append(c); nms.append(nm); arrived.append(int(info["arrived"]))
    print(f"    [{variant}] seed {seed}: {how}, path cost on full map={np.mean(costs):.3f}, "
          f"length={np.mean(nms):.1f} nm, arrived={np.mean(arrived)*100:.0f}%")
    return costs, nms, arrived


def summarise(results, baselines, out_dir, fig_dir):
    rows, seed_means = [], {}
    for v, r in results.items():
        s = pd.DataFrame({"seed": r["seed"], "c": r["path_costs_full_map"]}).groupby("seed")["c"].mean()
        seed_means[v] = s
    base = seed_means["V1: No Map"]
    for v, s in seed_means.items():
        n = len(s)
        h = stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n)
        t, p = stats.ttest_ind(s, base, equal_var=False) if v != "V1: No Map" else (np.nan, np.nan)
        rows.append({"Variant": v, "Path cost mean": round(s.mean(), 4), "SD (seeds)": round(s.std(ddof=1), 4),
                     "95% CI": f"[{s.mean() - h:.4f}, {s.mean() + h:.4f}]",
                     "Length nm": round(float(np.mean(results[v]["path_nm"])), 2),
                     "Arrival (%)": round(np.mean(results[v]["arrived"]) * 100, 1),
                     "vs V1 (%)": round((s.mean() - base.mean()) / base.mean() * 100, 1),
                     "Welch t vs V1": "" if v == "V1: No Map" else round(float(t), 2),
                     "p vs V1": "" if v == "V1: No Map" else f"{p:.2g}"})
    for b, c in baselines.items():
        rows.append({"Variant": f"{b} (full map)", "Path cost mean": round(c, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_pathcost.csv", index=False)
    df.to_csv(fig_dir / "ablation_table.csv", index=False)
    print("\n  All variants evaluated on the full AIS+ERA5 map (path-integrated cost, cost·nm):")
    print(df.to_string(index=False))

    # figure: mean ± 95% CI over seeds, baselines as lines; arrival rate
    vs = list(seed_means)
    means = [seed_means[v].mean() for v in vs]
    errs = [stats.t.ppf(0.975, len(seed_means[v]) - 1) * seed_means[v].std(ddof=1) / np.sqrt(len(seed_means[v])) for v in vs]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    x = np.arange(len(vs))
    ax.bar(x, means, yerr=errs, capsize=4, color=[COLORS[v] for v in vs], alpha=0.85)
    for b, ls in [("Dijkstra", "--"), ("Direct", ":")]:
        ax.axhline(baselines[b], color="#333", ls=ls, lw=1, label=f"{b} ({baselines[b]:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([v.split(": ")[0] + "\n" + v.split(": ")[1] for v in vs], fontsize=7)
    ax.set_ylabel("Path-integrated cost (cost·nm)")
    ax.set_title("Cost on the full map (mean ± 95% CI over seeds)", fontsize=8)
    ax.legend(fontsize=7)
    ax2.bar(x, [np.mean(results[v]["arrived"]) * 100 for v in vs], color=[COLORS[v] for v in vs], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Arrival rate (%)")
    ax2.set_title("Arrival rate", fontsize=8)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(fig_dir / f"fig_ablation{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    out_dir, fig_dir = Path(ABL_DIR), Path(FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("Building the cost maps...")
    variants = variant_maps()
    full_map = variants["V3: Full Map"][0]
    s, g = rx.latlon_to_ij(*rx.START), rx.latlon_to_ij(*rx.GOAL)
    baselines = {"Dijkstra": rx.path_cost_nm(rx.grid_path_latlons(rx.dijkstra(full_map, s, g)), full_map)[0],
                 "Direct":   rx.path_cost_nm(rx.grid_path_latlons(rx.direct_route(s, g)), full_map)[0]}
    results, t0 = {}, time.time()
    for v, (train_map, fields) in variants.items():
        print(f"\n  Variant: {v}")
        vdir = out_dir / variant_dir(v)
        vdir.mkdir(exist_ok=True)
        r = {"path_costs_full_map": [], "path_nm": [], "arrived": [], "seed": []}
        for seed in range(N_SEEDS):
            c, nm, a = train_and_eval(v, train_map, fields, seed, vdir, full_map)
            r["path_costs_full_map"] += c; r["path_nm"] += nm; r["arrived"] += a; r["seed"] += [seed] * len(c)
        results[v] = r
    with open(out_dir / "ablation_raw.json", "w") as f:
        json.dump({"variants": results, "baselines_full_map": baselines}, f, indent=2)
    summarise(results, baselines, out_dir, fig_dir)
    print(f"\n  ABLATION FINISHED in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
