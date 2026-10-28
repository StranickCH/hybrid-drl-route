"""
sac_maritime_baseline.py  (environment v3)
------------------------------------------
SAC-Discrete baseline (twin critics, fixed entropy coefficient alpha) in the same environment,
cost maps, scenarios, observation (with weather) and evaluation metric as run_experiments.py.

Same protocol as PPO except the training budget: N_SEEDS seeds × SAC_STEPS environment steps (SAC
converges within ~2e5 steps; PPO uses PPO_TIMESTEPS = 1e6), the best checkpoint by a deterministic
evaluation on separate start positions (seed * 1000 + 777) every EVAL_EVERY steps, final evaluation
on N_EVAL_EPISODES episodes (seed * 1000) with the path-integrated cost (cost·nm). One gradient
update every TRAIN_FREQ environment steps (as the default of DQN in Stable-Baselines3); a seed takes
~25 min on the CPU with four scenarios trained in parallel.

Usage:
  python sac_maritime_baseline.py all                   # 4 scenarios in parallel + summary
  python sac_maritime_baseline.py scenario "Normal"     # one scenario (used by "all")
  python sac_maritime_baseline.py summary               # summary of the finished scenarios
Output: sac_results/ (per-seed results, training curves, sac_vs_ppo.csv, figure)
Can be stopped and started again: finished seeds are reused.
"""
import os, sys, copy, json, time, random, subprocess, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as rx

OUT_DIR    = os.path.join(rx.ROOT, "sac_results")
ALPHA      = 0.05        # fixed entropy coefficient
LR, GAMMA, TAU = 3e-4, 0.99, 0.005
BATCH      = 256
BUFFER     = 200_000
WARMUP     = 10_000      # random actions before learning
TRAIN_FREQ = 4           # environment steps per gradient update
SAC_STEPS  = 200_000     # environment steps per seed; SAC converges within ~2e5 steps (PPO: 1e6)
THREADS    = 2
REPORT_EVERY = 60


def seed_file(scen, seed):
    return os.path.join(OUT_DIR, scen.replace(" ", "_"), f"seed{seed}.json")


# ── SAC-Discrete ────────────────────────────────────────────────────
def build_agent():
    import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

    def mlp(i, o, h=128):
        return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(),
                             nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, o))

    class Agent:
        def __init__(self):
            self.actor = mlp(18, 5)
            self.q1, self.q2 = mlp(18, 5), mlp(18, 5)
            self.q1t, self.q2t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
            self.oa = optim.Adam(self.actor.parameters(), lr=LR)
            self.oc = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)
            self.s = np.zeros((BUFFER, 18), np.float32); self.s2 = np.zeros((BUFFER, 18), np.float32)
            self.a = np.zeros((BUFFER, 1), np.int64); self.r = np.zeros((BUFFER, 1), np.float32)
            self.d = np.zeros((BUFFER, 1), np.float32); self.ptr = 0; self.size = 0

        def push(self, s, a, r, s2, d):
            i = self.ptr
            self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
            self.ptr = (i + 1) % BUFFER; self.size = min(self.size + 1, BUFFER)

        @torch.no_grad()
        def act(self, obs, det):
            p = F.softmax(self.actor(torch.as_tensor(obs).unsqueeze(0)), -1)
            return int(p.argmax(1)) if det else int(torch.distributions.Categorical(p).sample())

        def update(self):
            idx = np.random.randint(0, self.size, BATCH)
            s, a, r, s2, d = (torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))
            with torch.no_grad():
                p2 = F.softmax(self.actor(s2), -1); lp2 = torch.log(p2 + 1e-8)
                v2 = (p2 * (torch.min(self.q1t(s2), self.q2t(s2)) - ALPHA * lp2)).sum(1, keepdim=True)
                tgt = r + GAMMA * (1 - d) * v2
            lc = F.mse_loss(self.q1(s).gather(1, a), tgt) + F.mse_loss(self.q2(s).gather(1, a), tgt)
            self.oc.zero_grad(); lc.backward(); self.oc.step()
            p = F.softmax(self.actor(s), -1); lp = torch.log(p + 1e-8)
            with torch.no_grad():
                q = torch.min(self.q1(s), self.q2(s))
            la = (p * (ALPHA * lp - q)).sum(1).mean()
            self.oa.zero_grad(); la.backward(); self.oa.step()
            for net, tnet in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for w, wt in zip(net.parameters(), tnet.parameters()):
                    wt.data.mul_(1 - TAU).add_(TAU * w.data)

    return Agent()


def run_episodes(agent, cost_map, fields, env_seed, n):
    """Deterministic episodes: list of dicts with path cost (route completed to the goal), arrival, return."""
    env = rx.MaritimeEnv(cost_map, seed=env_seed, fields=fields)
    out = []
    for _ in range(n):
        obs, _ = env.reset(); done = False; ret = 0.0; info = {"arrived": False}
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, det=True))
            ret += r; done = term or trunc
        c, nm = rx.path_cost_nm(list(env.trajectory) + [rx.GOAL], cost_map)
        out.append({"path_cost": c, "path_nm": nm, "arrived": int(info["arrived"]), "return": ret})
    return out


def train_seed(scen, seed, cost_map, fields):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = build_agent()
    env = rx.MaritimeEnv(cost_map, seed=seed, fields=fields)
    obs, _ = env.reset()
    best, best_state, curve = -np.inf, None, []
    ep_ret, ep_rets = 0.0, []
    for step in range(1, SAC_STEPS + 1):
        a = env.action_space.sample() if step <= WARMUP else agent.act(obs, det=False)
        obs2, r, term, trunc, _ = env.step(a)
        agent.push(obs, a, r, obs2, float(term))
        obs = obs2; ep_ret += r
        if term or trunc:
            ep_rets.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()
        if step > WARMUP and step % TRAIN_FREQ == 0:
            agent.update()
        if step % rx.EVAL_EVERY == 0:
            ev = run_episodes(agent, cost_map, fields, seed * 1000 + 777, rx.N_EVAL_CKPT)
            mean_ret = float(np.mean([e["return"] for e in ev]))
            curve.append({"step": step, "eval_return": mean_ret,
                          "train_return_last100": float(np.mean(ep_rets[-100:])) if ep_rets else float("nan")})
            if mean_ret > best:
                best, best_state = mean_ret, copy.deepcopy(agent.actor.state_dict())
            with open(seed_file(scen, seed) + ".progress", "w") as f:
                f.write(f"{step} {curve[-1]['train_return_last100']:.1f}")
    if best_state is not None:
        agent.actor.load_state_dict(best_state)
    torch.save(agent.actor.state_dict(), seed_file(scen, seed).replace(".json", "_actor.pt"))
    final = run_episodes(agent, cost_map, fields, seed * 1000, rx.N_EVAL_EPISODES)
    return {"seed": seed, "episodes": final, "curve": curve, "best_eval_return": best}


def one_scenario(scen):
    import torch
    torch.set_num_threads(THREADS)
    os.makedirs(os.path.dirname(seed_file(scen, 0)), exist_ok=True)
    cost_map, fields = rx.build_cost_map(**rx.SCENARIOS[scen], return_fields=True)
    for seed in range(rx.N_SEEDS):
        if os.path.exists(seed_file(scen, seed)):
            continue
        t0 = time.time()
        res = train_seed(scen, seed, cost_map, fields)
        with open(seed_file(scen, seed), "w") as f:
            json.dump(res, f)
        c = np.mean([e["path_cost"] for e in res["episodes"]])
        a = np.mean([e["arrived"] for e in res["episodes"]]) * 100
        print(f"{scen}: seed {seed} path cost {c:.3f}, arrived {a:.0f}% [{time.time() - t0:.0f}s]", flush=True)


def progress(scen):
    done = sum(os.path.exists(seed_file(scen, s)) for s in range(rx.N_SEEDS))
    if done == rx.N_SEEDS:
        return f"done ({done}/{rx.N_SEEDS} seeds)"
    try:
        step, ret = open(seed_file(scen, done) + ".progress").read().split()
    except Exception:
        step, ret = "0", "nan"
    return (f"seed {done + 1}/{rx.N_SEEDS}: {int(step) / 1000:4.0f}k/{SAC_STEPS / 1000:.0f}k steps, "
            f"return (last 100 episodes) {float(ret):7.1f}")


def run_all():
    os.makedirs(os.path.join(rx.ROOT, "logs"), exist_ok=True)
    procs = []
    for scen in rx.SCENARIOS:
        log = open(os.path.join(rx.ROOT, "logs", f"sac_{scen.replace(' ', '_')}.log"), "w")
        procs.append((scen, subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "scenario", scen],
                                             stdout=log, stderr=subprocess.STDOUT), log))
    t0 = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(REPORT_EVERY)
        print(f"--- {time.strftime('%H:%M')}  ({(time.time() - t0) / 60:.0f} min since start)", flush=True)
        for scen, _, _ in procs:
            print(f"  {scen:16s} {progress(scen)}", flush=True)
    failed = [s for s, p, log in procs if p.wait() != 0]
    for _, _, log in procs:
        log.close()
    if failed:
        print("FAILED:", failed, "(see logs\\sac_*.log)")
        sys.exit(1)
    summary()


def summary():
    """SAC vs PPO (results/experiments/*/raw_results.json) vs Dijkstra; unit = seed."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, per_scen = [], {}
    for scen in rx.SCENARIOS:
        files = [seed_file(scen, s) for s in range(rx.N_SEEDS)]
        if not all(os.path.exists(f) for f in files):
            print(f"{scen}: not finished, skipped"); continue
        res = [json.load(open(f)) for f in files]
        sac = np.array([np.mean([e["path_cost"] for e in r["episodes"]]) for r in res])
        sac_arr = np.mean([e["arrived"] for r in res for e in r["episodes"]]) * 100
        raw = json.load(open(os.path.join(rx.EXP_DIR, scen.replace(" ", "_"), "raw_results.json")))
        ppo = pd.DataFrame({"s": raw["ppo_seed"], "c": raw["ppo_path_costs"]}).groupby("s")["c"].mean().values
        ppo_arr = np.mean(raw["ppo_arrived"]) * 100
        dij = raw["baseline_path_costs"]["Dijkstra"]
        per_scen[scen] = (sac, ppo, dij)
        u = stats.mannwhitneyu(sac, ppo)
        w = stats.wilcoxon(sac, ppo)   # same seeds and evaluation start positions
        for name, x, arr in (("SAC-Discrete", sac, sac_arr), ("PPO", ppo, ppo_arr)):
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({"Scenario": scen, "Algorithm": name, "Path cost mean": round(x.mean(), 4),
                         "95% CI": f"[{x.mean() - h:.4f}, {x.mean() + h:.4f}]",
                         "Median": round(float(np.median(x)), 4), "IQM": round(float(np.sort(x)[2:8].mean()), 4),
                         "Arrival (%)": round(arr, 1), "vs Dijkstra (%)": round((x.mean() / dij - 1) * 100, 1),
                         "Seeds within 25% of Dijkstra": int((x < 1.25 * dij).sum()),
                         "Mann-Whitney p (SAC vs PPO)": f"{u.pvalue:.3g}", "Wilcoxon p (SAC vs PPO)": f"{w.pvalue:.3g}"})
        rows.append({"Scenario": scen, "Algorithm": "Dijkstra", "Path cost mean": round(dij, 4)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "sac_vs_ppo.csv"), index=False)
    print(df.to_string(index=False))
    fig, axs = plt.subplots(1, len(per_scen), figsize=(3.2 * len(per_scen), 3.2), squeeze=False)
    for ax, (scen, (sac, ppo, dij)) in zip(axs[0], per_scen.items()):
        ax.boxplot([ppo, sac], widths=0.5)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["PPO", "SAC"])   # works with old and new matplotlib
        for k, x in enumerate((ppo, sac), 1):
            ax.plot(np.full(len(x), k) + np.random.uniform(-0.08, 0.08, len(x)), x, "o", ms=3, alpha=0.6)
        ax.axhline(dij, color="#333", ls="--", lw=1, label=f"Dijkstra ({dij:.2f})")
        ax.set_title(scen, fontsize=9); ax.legend(fontsize=7)
    axs[0][0].set_ylabel("Path-integrated cost per seed (cost·nm)")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_sac_vs_ppo{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}\\sac_vs_ppo.csv, fig_sac_vs_ppo.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT_DIR, exist_ok=True)
    if mode == "all":
        run_all()
    elif mode == "scenario":
        one_scenario(sys.argv[2])
    elif mode == "summary":
        summary()
