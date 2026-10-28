"""
make_paper_figures.py  (environment v3)
---------------------------------------
Figures and tables of the routing part of the paper, from the results of run_experiments.py,
ablation_study.py and (if present) sac_maritime_baseline.py. The unit of statistical analysis is
the training seed (mean path-integrated cost of its N_EVAL_EPISODES evaluation episodes).

  fig_cost_map.png/pdf          cost map (Normal) with the non-navigable cells
  fig_cost_components.png/pdf   fuel, risk and time components
  fig_training_curves.png/pdf   PPO episode return vs training steps, 10 seeds, 4 scenarios
  fig_routes.png/pdf            PPO routes of all seeds, Dijkstra and the direct line, 4 scenarios
  fig_scenarios.png/pdf         path cost per seed and gap to Dijkstra, 4 scenarios
  fig_ablation.png/pdf          ablation: path cost per seed on the full map, arrival rate
  table_scenarios.csv           PPO vs Dijkstra (mean, 95% t-CI, median, IQM, arrival, tests)
  table_ablation.csv            ablation (mean, median, IQM, arrival, rank tests between variants)

Usage:  python make_paper_figures.py      (output: results/figures/paper/)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import run_experiments as rx

OUT = Path(rx.ROOT) / "results" / "figures" / "paper"
ABL = Path(rx.ROOT) / "results" / "ablation" / "ablation_raw.json"
SCEN = list(rx.SCENARIOS)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
C_PPO, C_DIJ, C_DIR = "#1b9e77", "#1f4e9c", "#d95f02"
LAND = "#c8c8c8"


def save(fig, name):
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"{name}{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def iqm(x):
    x = np.sort(np.asarray(x)); k = len(x) // 4
    return float(x[k:len(x) - k].mean())


def tci(x):
    x = np.asarray(x); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def raw(scen):
    return json.load(open(Path(rx.EXP_DIR) / scen.replace(" ", "_") / "raw_results.json"))


def seed_means(r, key="ppo_path_costs"):
    return pd.DataFrame({"s": r["ppo_seed"], "v": r[key]}).groupby("s")["v"].mean().values


def grid_axes(ax):
    lats = np.linspace(rx.LAT_MIN, rx.LAT_MAX, rx.GRID_ROWS)
    lons = np.linspace(rx.LON_MIN, rx.LON_MAX, rx.GRID_COLS)
    return lons, lats


def draw_map(ax, field, cmap="RdYlGn_r", vmin=None, vmax=None):
    lons, lats = grid_axes(ax)
    im = ax.pcolormesh(lons, lats, np.ma.masked_where(~rx.NAV, field), cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.pcolormesh(lons, lats, np.ma.masked_where(rx.NAV, np.ones_like(field)), cmap=matplotlib.colors.ListedColormap([LAND]),
                  shading="auto")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.plot(rx.START[1], rx.START[0], "ko", ms=4); ax.plot(rx.GOAL[1], rx.GOAL[0], "k*", ms=7)
    return im


def rollout(model, cost_map, fields, seed):
    env = rx.MaritimeEnv(cost_map, seed=seed * 1000, fields=fields)
    obs, _ = env.reset(); done = False; info = {"arrived": False}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, info = env.step(int(a)); done = t or tr
    return list(env.trajectory), info["arrived"]


# ── cost map and its components ────────────────────────────────────
def fig_cost_map():
    cost, fields = rx.build_cost_map(**rx.SCENARIOS["Normal"], return_fields=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    im = draw_map(ax, cost, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Cell cost C(i, j)")
    ax.set_title("Cost map, Normal scenario (grey: non-navigable)")
    save(fig, "fig_cost_map")

    wind, wave, sog = fields[0] * 15.0, fields[1] * 3.0, fields[2] * 15.0
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for ax, f, lab in zip(axs, (wind, wave, sog), ("Wind speed (m/s)", "Significant wave height (m)", "Mean AIS SOG (kn)")):
        im = draw_map(ax, f, cmap="viridis"); fig.colorbar(im, ax=ax, label=lab)
    for ax in axs[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "fig_cost_components")


# ── training curves ────────────────────────────────────────────────
def fig_training_curves():
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0), sharey=True)
    for ax, scen in zip(axs, SCEN):
        d0 = Path(rx.EXP_DIR) / scen.replace(" ", "_")
        for s in range(rx.N_SEEDS):
            f = d0 / f"ppo_seed{s}_monitor.monitor.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, skiprows=1)
            ax.plot(d["l"].cumsum() / 1e6, d["r"].rolling(100, min_periods=10).mean(), lw=0.7, alpha=0.8)
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_title(scen); ax.set_xlabel("Training steps (M)")
    axs[0].set_ylabel("Episode return\n(moving mean, 100 episodes)")
    fig.tight_layout()
    save(fig, "fig_training_curves")


# ── routes ─────────────────────────────────────────────────────────
def fig_routes():
    from stable_baselines3 import PPO
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.6))
    s_ij, g_ij = rx.latlon_to_ij(*rx.START), rx.latlon_to_ij(*rx.GOAL)
    for ax, scen in zip(axs.flat, SCEN):
        cost, fields = rx.build_cost_map(**rx.SCENARIOS[scen], return_fields=True)
        im = draw_map(ax, cost, vmin=0, vmax=1)
        dij = rx.grid_path_latlons(rx.dijkstra(cost, s_ij, g_ij))
        dr = rx.grid_path_latlons(rx.direct_route(s_ij, g_ij))
        ax.plot([p[1] for p in dr], [p[0] for p in dr], color=C_DIR, ls="--", lw=1, label="Direct line (crosses land)")
        d0 = Path(rx.EXP_DIR) / scen.replace(" ", "_")
        for s in range(rx.N_SEEDS):
            pts, arrived = rollout(PPO.load(str(d0 / f"ppo_seed{s}.zip"), device="cpu"), cost, fields, s)
            ax.plot([p[1] for p in pts], [p[0] for p in pts], color=C_PPO, lw=0.8, alpha=0.8,
                    ls="-" if arrived else ":", label="PPO, 10 seeds" if s == 0 else None)
        ax.plot([p[1] for p in dij], [p[0] for p in dij], color=C_DIJ, lw=1.4, label="Dijkstra")
        ax.set_xlim(-95.4, -94.55); ax.set_ylim(29.2, 29.85)
        ax.set_title(scen)
    axs[0, 0].legend(loc="upper right")
    for ax in axs[0]:
        ax.set_xlabel("")
    for ax in axs[:, 1]:
        ax.set_ylabel("")
    fig.colorbar(im, ax=axs, shrink=0.6, label="Cell cost C(i, j)")
    save(fig, "fig_routes")


# ── scenario statistics ────────────────────────────────────────────
def table_and_fig_scenarios():
    rows, data = [], {}
    for scen in SCEN:
        r = raw(scen)
        x = seed_means(r); dij = r["baseline_path_costs"]["Dijkstra"]
        arr = seed_means(r, "ppo_arrived") * 100
        lo, hi = tci(x)
        w = stats.wilcoxon(x - dij)
        data[scen] = (x, dij, arr)
        rows.append({"Scenario": scen, "Dijkstra": round(dij, 3), "PPO mean": round(x.mean(), 3),
                     "PPO 95% CI": f"[{lo:.3f}, {hi:.3f}]", "PPO median": round(float(np.median(x)), 3),
                     "PPO IQM": round(iqm(x), 3), "PPO SD (seeds)": round(x.std(ddof=1), 3),
                     "Gap mean (%)": round((x.mean() / dij - 1) * 100, 1),
                     "Gap median (%)": round((np.median(x) / dij - 1) * 100, 1),
                     "Arrival (%)": round(float(arr.mean()), 1),
                     "Seeds within 10% of Dijkstra": int((x <= 1.10 * dij).sum()),
                     "Wilcoxon p (PPO vs Dijkstra)": f"{w.pvalue:.3g}",
                     "Direct line (crosses land)": round(r["baseline_path_costs"]["Direct"], 2)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "table_scenarios.csv", index=False)
    print(df.to_string(index=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for k, scen in enumerate(SCEN):
        x, dij, arr = data[scen]
        a1.boxplot([x], positions=[k], widths=0.5, showfliers=False)
        a1.plot(np.full(len(x), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(x)), x, "o",
                color=C_PPO, ms=3, alpha=0.8)
        a1.plot([k - 0.35, k + 0.35], [dij, dij], color=C_DIJ, lw=1.5, label="Dijkstra" if k == 0 else None)
        gap = (x / dij - 1) * 100
        a2.boxplot([gap], positions=[k], widths=0.5, showfliers=False)
        a2.plot(np.full(len(gap), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(gap)), gap, "o",
                color=C_PPO, ms=3, alpha=0.8)
    for a in (a1, a2):
        a.set_xticks(range(len(SCEN))); a.set_xticklabels([s.replace(" ", "\n") for s in SCEN])
    a1.set_ylabel("Path-integrated cost per seed (cost·nm)"); a1.legend(loc="upper left")
    a2.set_ylabel("Gap to Dijkstra (%)"); a2.axhline(0, color=C_DIJ, lw=0.8)
    a1.set_title("(a) PPO path cost, 10 seeds"); a2.set_title("(b) Gap to the Dijkstra route")
    fig.tight_layout()
    save(fig, "fig_scenarios")


# ── ablation ───────────────────────────────────────────────────────
def table_and_fig_ablation():
    if not ABL.exists():
        print("no ablation results"); return
    a = json.load(open(ABL)); dij = a["baselines_full_map"]["Dijkstra"]
    S, A = {}, {}
    for v, d in a["variants"].items():
        df = pd.DataFrame({"s": d["seed"], "c": d["path_costs_full_map"], "a": d["arrived"]})
        S[v] = df.groupby("s")["c"].mean().values; A[v] = df["a"].mean() * 100
    names = list(S)
    rows = []
    for v in names:
        x = S[v]
        rows.append({"Variant": v, "Mean": round(x.mean(), 3), "Median": round(float(np.median(x)), 3),
                     "IQM": round(iqm(x), 3), "Arrival (%)": round(A[v], 1),
                     "Gap median to Dijkstra (%)": round((np.median(x) / dij - 1) * 100, 1)})
    pairs = [("V2: Traffic Map", "V1: No Map"), ("V3: Full Map", "V1: No Map"), ("V3: Full Map", "V2: Traffic Map"),
             ("V4: Full Map + State", "V3: Full Map"), ("V4: Full Map + State", "V1: No Map")]
    tests = [{"Comparison": f"{p} vs {q}", "Wilcoxon p (paired by seed)": f"{stats.wilcoxon(S[p], S[q]).pvalue:.3g}",
              "Mann-Whitney p": f"{stats.mannwhitneyu(S[p], S[q]).pvalue:.3g}"} for p, q in pairs if p in S and q in S]
    pd.DataFrame(rows).to_csv(OUT / "table_ablation.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "table_ablation_tests.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(tests).to_string(index=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [2, 1]})
    for k, v in enumerate(names):
        x = S[v]
        a1.boxplot([x], positions=[k], widths=0.5, showfliers=False)
        a1.plot(np.full(len(x), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(x)), x, "o", color=C_PPO, ms=3)
    a1.axhline(dij, color=C_DIJ, lw=1, label=f"Dijkstra ({dij:.2f})")
    hi = max(np.percentile(np.concatenate(list(S.values())), 95) * 1.15, dij * 1.6)
    a1.set_ylim(dij * 0.9, hi)
    for k, v in enumerate(names):   # outliers above the axis
        n_out = int((S[v] > hi).sum())
        if n_out:
            a1.annotate(f"{n_out} seed(s) > {hi:.1f}", (k, hi), xytext=(k, hi * 0.97), ha="center", fontsize=6)
    labels = [v.split(": ")[0] + "\n" + v.split(": ")[1] for v in names]
    a1.set_xticks(range(len(names))); a1.set_xticklabels(labels)
    a1.set_ylabel("Path cost on the full map (cost·nm)"); a1.legend(loc="upper right")
    a1.set_title("(a) Path cost per seed")
    a2.bar(range(len(names)), [A[v] for v in names], color=C_PPO, alpha=0.8)
    a2.set_xticks(range(len(names))); a2.set_xticklabels([v.split(": ")[0] for v in names])
    a2.set_ylim(0, 105); a2.set_ylabel("Arrival rate (%)"); a2.set_title("(b) Arrival rate")
    fig.tight_layout()
    save(fig, "fig_ablation")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_cost_map()
    fig_training_curves()
    table_and_fig_scenarios()
    table_and_fig_ablation()
    fig_routes()
