"""
train_parallel.py
-----------------
Trains the PPO seeds of all scenarios (or of all ablation variants) in parallel processes,
one process per scenario/variant, each limited to THREADS torch threads. Every REPORT_EVERY
seconds the progress of each process is printed (each process writes its own log to logs/).
Seeds whose model already exists are skipped, so the script can be stopped and started again.
run_experiments.py and ablation_study.py then find all models and only evaluate them.

Usage:
  python train_parallel.py experiments     # 4 scenarios × N_SEEDS
  python train_parallel.py ablation        # 4 ablation variants × N_SEEDS
  python train_parallel.py one-scenario "Normal"        (used internally)
  python train_parallel.py one-variant "V1: No Map"     (used internally)
"""
import subprocess
import sys
import time
from pathlib import Path

THREADS = 2
REPORT_EVERY = 60   # seconds between progress lines in the console


def progress(folder, prefix, n_seeds, total_steps):
    """Progress of one scenario/variant from its files: trained seeds and the running seed."""
    done = sum((folder / f"{prefix}{s}.zip").exists() for s in range(n_seeds))
    if done == n_seeds:
        return f"done ({n_seeds}/{n_seeds} seeds)"
    mon = folder / f"{prefix}{done}_monitor.monitor.csv"
    steps, ret = 0, float("nan")
    try:
        import pandas as pd
        d = pd.read_csv(mon, skiprows=1)
        steps, ret = int(d["l"].sum()), d["r"].tail(100).mean()
    except Exception:
        pass
    return (f"seed {done + 1}/{n_seeds}: {steps / 1000:4.0f}k/{total_steps / 1000:.0f}k steps, "
            f"return (last 100 episodes) {ret:7.1f}")


def one_scenario(name):
    import torch
    torch.set_num_threads(THREADS)
    import run_experiments as rx
    cost_map, fields = rx.build_cost_map(**rx.SCENARIOS[name], return_fields=True)
    scen_dir = Path(rx.EXP_DIR) / name.replace(" ", "_")
    scen_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(rx.N_SEEDS):
        path = scen_dir / f"ppo_seed{seed}.zip"
        if path.exists():
            continue
        t0 = time.time()
        rx.train_ppo_seed(cost_map, seed, rx.PPO_TIMESTEPS, path, fields=fields)
        print(f"{name}: seed {seed} trained [{time.time() - t0:.0f}s]", flush=True)


def one_variant(name):
    import torch
    torch.set_num_threads(THREADS)
    import ablation_study as ab
    import run_experiments as rx
    train_map, fields = ab.variant_maps()[name]
    vdir = Path(ab.ABL_DIR) / ab.variant_dir(name)
    vdir.mkdir(parents=True, exist_ok=True)
    for seed in range(ab.N_SEEDS):
        path = vdir / f"model_seed{seed}.zip"
        if path.exists():
            continue
        t0 = time.time()
        rx.train_ppo_seed(train_map, seed, ab.PPO_STEPS, path, fields=fields)
        print(f"{name}: seed {seed} trained [{time.time() - t0:.0f}s]", flush=True)


def launch(mode, names, folders, prefix, n_seeds, total_steps):
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    procs = []
    for n in names:
        log = open(logs / f"train_{n.replace(' ', '_').replace(':', '').replace('+', 'plus')}.log", "w")
        procs.append((n, subprocess.Popen([sys.executable, "-u", __file__, mode, n], stdout=log, stderr=subprocess.STDOUT), log))
    t0 = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(REPORT_EVERY)
        print(f"--- {time.strftime('%H:%M')}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        for n, f in zip(names, folders):
            print(f"  {n:22s} {progress(f, prefix, n_seeds, total_steps)}", flush=True)
    failed = []
    for n, p, log in procs:
        if p.wait() != 0:
            failed.append(n)
        log.close()
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("all trained")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "experiments":
        import run_experiments as rx
        names = list(rx.SCENARIOS)
        launch("one-scenario", names, [Path(rx.EXP_DIR) / n.replace(" ", "_") for n in names],
               "ppo_seed", rx.N_SEEDS, rx.PPO_TIMESTEPS)
    elif mode == "ablation":
        import ablation_study as ab
        launch("one-variant", ab.VARIANTS, [Path(ab.ABL_DIR) / ab.variant_dir(v) for v in ab.VARIANTS],
               "model_seed", ab.N_SEEDS, ab.PPO_STEPS)
    elif mode == "one-scenario":
        one_scenario(sys.argv[2])
    elif mode == "one-variant":
        one_variant(sys.argv[2])
