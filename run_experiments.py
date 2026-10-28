"""
run_experiments.py
==================
Multi-seed experiment runner (PPO vs Dijkstra, A* and the direct route).

What it does:
  1. N_SEEDS PPO runs with different seeds — mean ± std
  2. 4 scenarios: Normal / Storm / Dense Traffic / Sparse Traffic
  3. t-test PPO vs every baseline
  4. Bootstrap confidence intervals (95%)
  5. All results → results/experiments/

Usage:
    python train_parallel.py experiments   # trains the 40 PPO seeds in 4 parallel processes (~4 h)
    python run_experiments.py              # evaluation and statistics (trains missing seeds itself)

Parameters are set in the CONFIG section below.
"""

import os, warnings, sys, time, json, itertools
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

AIS_MERGED  = os.path.join(ROOT, "ais_merged")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
EXP_DIR     = os.path.join(ROOT, "results", "experiments")

# Experiments
N_SEEDS          = 10       # number of PPO runs (for mean±std)
PPO_TIMESTEPS    = 1_000_000  # steps per seed (v2: 300_000; with land the start is harder to learn)
EVAL_EVERY       = 20_000   # environment steps between evaluations for checkpoint selection
N_EVAL_CKPT      = 5        # episodes per checkpoint evaluation
N_EVAL_EPISODES  = 30       # evaluation episodes per seed
BOOTSTRAP_N      = 2_000    # bootstrap iterations for the CI

# Scenarios: name → cost map modifier
# Environment v3 (2026-10-20): land is not navigable. navigable_mask.npy (make_navigable_mask.py):
# water in the global 1-km land mask or cells with AIS traffic (plus one cell), connected to
# START and GOAL. A move into a land cell is blocked: the vessel stays where it was and gets
# R_GROUNDING (the episode goes on, so the agent can learn to turn away). Dijkstra/A* do not
# enter land; land cells cost 1.0 (the maximum) in the evaluation metric, so the direct route,
# which crosses land, is a reference only. Turns of 30° per step (TURN_DEG). The best checkpoint
# of each seed (evaluation on a separate set of start positions every EVAL_EVERY steps) is kept
# instead of the last one.
_NAV_PATH = os.path.join(ROOT, "navigable_mask.npy")
if os.path.exists(_NAV_PATH):
    NAV = np.load(_NAV_PATH)
else:
    print("WARNING: navigable_mask.npy not found, all cells are navigable (environment v2)")
    NAV = None

SCENARIOS = {
    "Normal":       dict(storm_factor=1.0,  traffic_factor=1.0),
    "Storm":        dict(storm_factor=2.5,  traffic_factor=1.0),
    "Dense Traffic":dict(storm_factor=1.0,  traffic_factor=2.0),
    "Sparse Traffic":dict(storm_factor=1.0, traffic_factor=0.3),
}

# Grid
LAT_MIN, LAT_MAX = 29.0,  30.2
LON_MIN, LON_MAX = -95.5, -94.0
GRID_ROWS, GRID_COLS = 60, 80

START = (29.73, -95.27)
GOAL  = (29.31, -94.79)
V_MIN, V_MAX, V_INIT = 4.0, 18.0, 12.0
TURN_DEG = 30.0   # per 0.1-h step (v1/v2: 10°, chosen for 0.5-h steps; too slow to follow a channel)
R_ARRIVAL, R_COLLISION = 100.0, -50.0
R_GROUNDING = -5.0   # v3: a move into a land cell is blocked (the vessel stays) and penalised
# Environment v2 (2026-10): the reward is aligned with the evaluation metric.
#   r_t = -W_COST * (path-integrated cost of the step, cost·nm) + W_PROGRESS * (d_{t-1} - d_t)
#         + R_ARRIVAL on arrival, R_COLLISION when leaving the grid.
# The progress term sums to the same value on every route that reaches the goal, so the
# agent minimises the same path-integrated cost as Dijkstra/A*. (v1 used -cell cost - 0.1*distance
# to the goal per step with 0.5 h steps; the distance term dominated and the agent sailed straight.)
DT_HOURS     = 0.1     # one step ≈ one grid cell at 12 kn (v1: 0.5 h ≈ 5 cells)
W_COST       = 10.0
W_PROGRESS   = 1.0     # per nautical mile of progress towards the goal
ARRIVAL_NM   = 1.5
MAX_STEPS    = 300
STEP_DS_NM   = 0.2     # sampling of the cost along one step (training)
N_ENVS = 4
WEIGHTS = dict(alpha=0.4, beta=0.4, gamma=0.2)

# Colours
COLORS = {
    "PPO":          "#2ecc71",
    "A*":           "#e74c3c",
    "Dijkstra":     "#3498db",
    "Direct":       "#f39c12",
}
SCENARIO_COLORS = {
    "Normal":        "#2c3e50",
    "Storm":         "#e74c3c",
    "Dense Traffic": "#9b59b6",
    "Sparse Traffic":"#27ae60",
}


# ════════════════════════════════════════════════════════════════════
#  1. COST MAP with scenario support
# ════════════════════════════════════════════════════════════════════

def build_cost_map(storm_factor: float = 1.0,
                   traffic_factor: float = 1.0, return_fields: bool = False):
    """
    Builds the cost map. storm_factor > 1 amplifies the waves,
    traffic_factor scales the traffic density (SOG).
    """
    files = sorted(Path(AIS_MERGED).glob("*_era5.parquet"))[:80]
    sum_wind_sp = np.zeros((GRID_ROWS, GRID_COLS))
    sum_wave_hs = np.zeros((GRID_ROWS, GRID_COLS))
    sum_sog     = np.zeros((GRID_ROWS, GRID_COLS))
    counts      = {k: np.zeros((GRID_ROWS, GRID_COLS)) for k in ("wind_speed", "wave_hs", "SOG")}

    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "LAT" not in df.columns:
            continue
        mask = (df["LAT"].between(LAT_MIN, LAT_MAX) &
                df["LON"].between(LON_MIN, LON_MAX))
        df = df[mask]
        if df.empty:
            continue
        i_arr = ((df["LAT"] - LAT_MIN) / (LAT_MAX - LAT_MIN) *
                 (GRID_ROWS-1)).astype(int).clip(0, GRID_ROWS-1)
        j_arr = ((df["LON"] - LON_MIN) / (LON_MAX - LON_MIN) *
                 (GRID_COLS-1)).astype(int).clip(0, GRID_COLS-1)
        for src, dst in [("wind_speed", sum_wind_sp),
                         ("wave_hs",    sum_wave_hs),
                         ("SOG",        sum_sog)]:
            if src in df.columns:
                vals = df[src].values.astype(float)
                ok = ~np.isnan(vals)   # missing values are not counted (they used to be counted as 0)
                np.add.at(dst, (i_arr.values[ok], j_arr.values[ok]), vals[ok])
                np.add.at(counts[src], (i_arr.values[ok], j_arr.values[ok]), 1)

    def avg(arr, cnt):
        m = cnt > 0
        a = np.zeros_like(arr)
        a[m] = arr[m] / cnt[m]
        return gaussian_filter(a, sigma=1.5)

    wind_sp = avg(sum_wind_sp, counts["wind_speed"])
    wave_hs = avg(sum_wave_hs, counts["wave_hs"]) * storm_factor      # Storm scenario
    sog     = avg(sum_sog, counts["SOG"])         * traffic_factor    # Traffic scenario

    def norm(a):
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-9 else np.zeros_like(a)

    f_fuel = norm(wind_sp ** 2)
    f_risk = norm(1.0 / (1.0 + np.exp(-4.0 * (wave_hs - 1.5))))
    f_time = norm(12.0 / np.maximum(sog, 0.5))

    cost = (WEIGHTS["alpha"] * f_fuel +
            WEIGHTS["beta"]  * f_risk +
            WEIGHTS["gamma"] * f_time)
    cost = norm(cost).astype(np.float32)
    if NAV is not None:
        cost[~NAV] = 1.0          # land: maximum cost in the metric (the agent cannot enter it)
    if return_fields:
        # local fields for the observation (same smoothing and scenario factors as the map)
        fields = np.stack([wind_sp / 15.0, wave_hs / 3.0, sog / 15.0]).astype(np.float32)
        return cost, fields
    return cost


# ════════════════════════════════════════════════════════════════════
#  2. NAVIGATION (helper functions)
# ════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.radians(lat2))
    x = (np.cos(np.radians(lat1))*np.sin(np.radians(lat2)) -
         np.sin(np.radians(lat1))*np.cos(np.radians(lat2))*np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def latlon_to_ij(lat, lon):
    i = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (GRID_ROWS - 1))
    j = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_COLS - 1))
    return np.clip(i, 0, GRID_ROWS-1), np.clip(j, 0, GRID_COLS-1)

def ij_to_latlon(i, j):
    lat = LAT_MIN + i / (GRID_ROWS - 1) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + j / (GRID_COLS - 1) * (LON_MAX - LON_MIN)
    return lat, lon

def neighbors_8(i, j):
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID_ROWS and 0 <= nj < GRID_COLS and (NAV is None or NAV[ni, nj]):
                step = 1.414 if abs(di) + abs(dj) == 2 else 1.0
                yield ni, nj, step

def route_cost(path_ij, cost_map):
    return float(sum(cost_map[i, j] for i, j in path_ij))

def route_distance_nm(path_ij):
    total = 0.0
    for k in range(1, len(path_ij)):
        la1, lo1 = ij_to_latlon(*path_ij[k-1])
        la2, lo2 = ij_to_latlon(*path_ij[k])
        total += haversine(la1, lo1, la2, lo2)
    return total

def path_cost_nm(latlons, cost_map, ds_nm=0.05):
    """Path-integrated cost: cell cost × distance along a polyline (cost·nm), sampled every ds_nm.
    The same metric for PPO trajectories and grid routes, independent of the number of steps
    (route_cost sums one value per grid cell or per PPO step). Returns (cost, length_nm)."""
    total, length = 0.0, 0.0
    for (la1, lo1), (la2, lo2) in zip(latlons[:-1], latlons[1:]):
        d = haversine(la1, lo1, la2, lo2)
        n = max(1, int(np.ceil(d / ds_nm)))
        t = (np.arange(n) + 0.5) / n
        i = np.clip(((la1 + t * (la2 - la1) - LAT_MIN) / (LAT_MAX - LAT_MIN) * (GRID_ROWS - 1)).astype(int), 0, GRID_ROWS - 1)
        j = np.clip(((lo1 + t * (lo2 - lo1) - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_COLS - 1)).astype(int), 0, GRID_COLS - 1)
        total += float(cost_map[i, j].sum()) * d / n      # same sampling as latlon_to_ij, vectorised
        length += d
    return float(total), float(length)

def grid_path_latlons(path_ij):
    """Grid route as a polyline START → cell centres → GOAL."""
    def centre(i, j):
        return (LAT_MIN + (i + 0.5) / (GRID_ROWS - 1) * (LAT_MAX - LAT_MIN),
                LON_MIN + (j + 0.5) / (GRID_COLS - 1) * (LON_MAX - LON_MIN))
    return [START] + [centre(i, j) for i, j in path_ij[1:-1]] + [GOAL]


# ════════════════════════════════════════════════════════════════════
#  3. BASELINE ALGORITHMS (Dijkstra, A*, Direct)
# ════════════════════════════════════════════════════════════════════

import heapq

def dijkstra(cost_map, start_ij, goal_ij):
    si, sj = start_ij
    gi, gj = goal_ij
    dist = np.full((GRID_ROWS, GRID_COLS), np.inf)
    dist[si, sj] = 0.0
    prev = {}
    heap = [(0.0, si, sj)]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        if (i, j) == (gi, gj):
            break
        for ni, nj, step in neighbors_8(i, j):
            cell_cost = cost_map[ni, nj] + 0.01
            new_d = dist[i, j] + step * cell_cost
            if new_d < dist[ni, nj]:
                dist[ni, nj] = new_d
                prev[(ni, nj)] = (i, j)
                heapq.heappush(heap, (new_d, ni, nj))
    path = []
    cur = (gi, gj)
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append((si, sj))
    path.reverse()
    return path

def astar(cost_map, start_ij, goal_ij):
    si, sj = start_ij
    gi, gj = goal_ij
    def h(i, j):
        return np.sqrt((i - gi)**2 + (j - gj)**2) * cost_map.min()
    g = np.full((GRID_ROWS, GRID_COLS), np.inf)
    g[si, sj] = 0.0
    prev = {}
    heap = [(h(si, sj), 0.0, si, sj)]
    while heap:
        f, gv, i, j = heapq.heappop(heap)
        if gv > g[i, j]:
            continue
        if (i, j) == (gi, gj):
            break
        for ni, nj, step in neighbors_8(i, j):
            cell_cost = cost_map[ni, nj] + 0.01
            ng = g[i, j] + step * cell_cost
            if ng < g[ni, nj]:
                g[ni, nj] = ng
                prev[(ni, nj)] = (i, j)
                heapq.heappush(heap, (ng + h(ni, nj), ng, ni, nj))
    path = []
    cur = (gi, gj)
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append((si, sj))
    path.reverse()
    return path

def direct_route(start_ij, goal_ij):
    si, sj = start_ij
    gi, gj = goal_ij
    steps = max(abs(gi - si), abs(gj - sj))
    if steps == 0:
        return [start_ij]
    path = []
    for k in range(steps + 1):
        t = k / steps
        i = int(round(si + t * (gi - si)))
        j = int(round(sj + t * (gj - sj)))
        path.append((np.clip(i, 0, GRID_ROWS-1),
                     np.clip(j, 0, GRID_COLS-1)))
    return path


# ════════════════════════════════════════════════════════════════════
#  4. GYMNASIUM ENVIRONMENT
# ════════════════════════════════════════════════════════════════════

import gymnasium as gym
from gymnasium import spaces

class MaritimeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cost_map, seed=None, fields=None):
        """fields: array (3, GRID_ROWS, GRID_COLS) with wind speed, wave height and traffic speed
        (normalised) seen by the agent in its cell; None = no weather information (zeros)."""
        super().__init__()
        self.cost_map = cost_map
        self.fields = fields
        self._np_random = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(5)
        low  = np.full(18, -np.inf, dtype=np.float32)
        high = np.full(18,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        while True:   # random start near START, in a navigable cell
            self.lat  = START[0] + rng.uniform(-0.02, 0.02)
            self.lon  = START[1] + rng.uniform(-0.02, 0.02)
            if NAV is None or NAV[latlon_to_ij(self.lat, self.lon)]:
                break
        self.cog  = bearing(self.lat, self.lon, GOAL[0], GOAL[1])
        self.sog  = V_INIT + rng.uniform(-2, 2)
        self.step_n = 0
        self.max_steps = MAX_STEPS
        self.trajectory = [(self.lat, self.lon)]
        self.dist = haversine(self.lat, self.lon, GOAL[0], GOAL[1])
        return self._get_obs(), {}

    def _get_obs(self):
        dist = haversine(self.lat, self.lon, GOAL[0], GOAL[1])
        brng = bearing(self.lat, self.lon, GOAL[0], GOAL[1])
        lat_n  = (self.lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
        lon_n  = (self.lon - LON_MIN) / (LON_MAX - LON_MIN)
        cog_n  = self.cog / 360.0
        sog_n  = (self.sog - V_MIN) / (V_MAX - V_MIN)
        dist_n = dist / 200.0
        brng_n = brng  / 360.0
        i, j = latlon_to_ij(self.lat, self.lon)
        patch = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ii = np.clip(i+di, 0, GRID_ROWS-1)
                jj = np.clip(j+dj, 0, GRID_COLS-1)
                patch.append(self.cost_map[ii, jj] if NAV is None or NAV[ii, jj] else -1.0)   # -1 = land
        if self.fields is not None:
            ci, cj = latlon_to_ij(self.lat, self.lon)
            wx = [float(v) for v in self.fields[:, ci, cj]]   # wind speed, wave height, traffic speed
        else:
            wx = [0.0, 0.0, 0.0]
        return np.array([lat_n, lon_n, sog_n, cog_n,
                         *wx,
                         dist_n, brng_n, *patch], dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.cog = (self.cog - TURN_DEG) % 360
        elif action == 2:
            self.cog = (self.cog + TURN_DEG) % 360
        elif action == 3:
            self.sog = max(V_MIN, self.sog - 1.0)
        elif action == 4:
            self.sog = min(V_MAX, self.sog + 1.0)

        dist_nm = self.sog * DT_HOURS
        dlat = dist_nm * np.cos(np.radians(self.cog)) / 60.0
        dlon = dist_nm * np.sin(np.radians(self.cog)) / (
            60.0 * np.cos(np.radians(self.lat)))
        prev = (self.lat, self.lon)
        self.step_n += 1
        new_lat, new_lon = self.lat + dlat, self.lon + dlon
        grounded = (NAV is not None and LAT_MIN <= new_lat <= LAT_MAX and LON_MIN <= new_lon <= LON_MAX
                    and not NAV[latlon_to_ij(new_lat, new_lon)])
        if grounded:   # blocked move: the vessel stays in its cell
            info = {"cost": 0.0, "dist": self.dist, "arrived": False, "grounded": True}
            return (self._get_obs(), R_GROUNDING, False, self.step_n >= self.max_steps, info)
        self.lat, self.lon = new_lat, new_lon
        self.trajectory.append((self.lat, self.lon))

        cost = path_cost_nm([prev, (self.lat, self.lon)], self.cost_map, ds_nm=STEP_DS_NM)[0]   # cost·nm of the step
        dist = haversine(self.lat, self.lon, GOAL[0], GOAL[1])
        progress = self.dist - dist
        self.dist = dist

        out_of_bounds = not (LAT_MIN <= self.lat <= LAT_MAX and
                             LON_MIN <= self.lon <= LON_MAX)
        arrived = dist < ARRIVAL_NM

        if out_of_bounds:
            reward = R_COLLISION
        else:
            reward = -W_COST * cost + W_PROGRESS * progress + (R_ARRIVAL if arrived else 0.0)

        terminated = arrived or out_of_bounds
        truncated  = self.step_n >= self.max_steps

        info = {"cost": cost, "dist": dist, "arrived": arrived, "grounded": False}
        return self._get_obs(), reward, terminated, truncated, info


# ════════════════════════════════════════════════════════════════════
#  5. PPO TRAINING (one seed)
# ════════════════════════════════════════════════════════════════════

def train_ppo_seed(cost_map, seed: int, timesteps: int, save_path: Path, fields=None):
    """Trains PPO with the given seed and saves the model; episode rewards and lengths
    of the training run are written to <model>_monitor.csv (training curve)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)

    def make_env(rank):
        def _init():
            env = MaritimeEnv(cost_map, seed=seed + rank, fields=fields)
            return env
        return _init

    # DummyVecEnv: the environment is cheap, so worker processes only add overhead
    vec_env = VecMonitor(DummyVecEnv([make_env(i) for i in range(N_ENVS)]),
                         filename=str(save_path.with_suffix("")) + "_monitor")

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=0,
        seed=seed,
        tensorboard_log=None,
        device="cpu",  # small MLP: faster on the CPU than on the GPU
    )
    # checkpoint selection: deterministic evaluation on start positions not used in training
    # or in the final evaluation (seed * 1000)
    from stable_baselines3.common.callbacks import EvalCallback
    best_dir = save_path.parent / (save_path.stem + "_best")
    eval_cb = EvalCallback(MaritimeEnv(cost_map, seed=seed * 1000 + 777, fields=fields),
                           n_eval_episodes=N_EVAL_CKPT, eval_freq=max(1, EVAL_EVERY // N_ENVS),
                           best_model_save_path=str(best_dir), log_path=str(best_dir),
                           deterministic=True, verbose=0)
    model.learn(total_timesteps=timesteps, progress_bar=False, callback=eval_cb)
    model.save(str(save_path.parent / (save_path.stem + "_last")))
    vec_env.close()
    if (best_dir / "best_model.zip").exists():
        model = PPO.load(str(best_dir / "best_model.zip"), device="cpu")
    model.save(str(save_path))
    return model


def evaluate_ppo(model, cost_map, n_episodes: int, seed: int, fields=None):
    """Evaluates PPO and returns a list of per-episode metrics."""
    env = MaritimeEnv(cost_map, seed=seed * 1000, fields=fields)
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward, total_cost, steps = 0.0, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            total_cost   += info["cost"]
            steps += 1
            done = terminated or truncated
        p_cost, p_nm = path_cost_nm(list(env.trajectory) + [GOAL], cost_map)   # route completed to the goal
        results.append({
            "path_cost":  p_cost,
            "path_nm":    p_nm,
            "reward":     total_reward,
            "total_cost": total_cost,
            "steps":      steps,
            "arrived":    int(info["arrived"]),
            "final_dist": info["dist"],
        })
    return results


# ════════════════════════════════════════════════════════════════════
#  6. STATISTICS: bootstrap CI and t-test
# ════════════════════════════════════════════════════════════════════

def bootstrap_ci(data, n=BOOTSTRAP_N, ci=0.95):
    """Bootstrap 95% CI of the mean."""
    data = np.array(data)
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                  for _ in range(n)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(data), lo, hi

def ttest_vs_baseline(ppo_costs, baseline_cost_scalar, name):
    """
    t-test: PPO over many episodes vs a baseline (deterministic scalar).
    A one-sample t-test against the baseline value as μ0 is used.
    """
    t_stat, p_val = stats.ttest_1samp(ppo_costs, baseline_cost_scalar)
    return {
        "vs": name,
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "significant": p_val < 0.05,
        "ppo_mean": round(float(np.mean(ppo_costs)), 4),
        "baseline":  round(float(baseline_cost_scalar), 4),
    }


# ════════════════════════════════════════════════════════════════════
#  7. MAIN EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════════

def run_all_experiments():
    from stable_baselines3 import PPO

    exp_dir  = Path(EXP_DIR)
    fig_dir  = Path(FIGURES_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    start_ij = latlon_to_ij(*START)
    goal_ij  = latlon_to_ij(*GOAL)

    # ── Result storage ──
    all_results = {}   # {scenario: {method: [cost_per_episode×seeds]}}

    for scen_name, scen_params in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scen_name}")
        print(f"{'='*60}")

        cost_map, fields = build_cost_map(**scen_params, return_fields=True)

        # ── Baselines (deterministic) ──
        path_dijk   = dijkstra(cost_map, start_ij, goal_ij)
        path_astar  = astar(cost_map, start_ij, goal_ij)
        path_direct = direct_route(start_ij, goal_ij)

        # v1 cell-sum metric (one cell cost per grid cell), kept for reference only
        baseline_costs_cells = {
            "Dijkstra": route_cost(path_dijk,   cost_map),
            "A*":       route_cost(path_astar,  cost_map),
            "Direct":   route_cost(path_direct, cost_map),
        }
        baseline_path = {name: path_cost_nm(grid_path_latlons(p), cost_map)
                         for name, p in [("Dijkstra", path_dijk), ("A*", path_astar), ("Direct", path_direct)]}
        # main metric: path-integrated cost (cost·nm), the same for all methods
        baseline_costs = {k: v[0] for k, v in baseline_path.items()}
        print(f"  Baselines, path cost (cost·nm): {baseline_costs}")

        # ── PPO × N_SEEDS ──
        ppo_all_costs   = []
        ppo_all_rewards = []
        ppo_all_arrived = []
        ppo_all_steps   = []
        ppo_all_pcost, ppo_all_pnm, ppo_all_seed = [], [], []

        scen_dir = exp_dir / scen_name.replace(" ", "_")
        scen_dir.mkdir(exist_ok=True)

        for seed in range(N_SEEDS):
            seed_path = scen_dir / f"ppo_seed{seed}.zip"
            print(f"  [Seed {seed+1}/{N_SEEDS}] training...", end=" ", flush=True)
            t0 = time.time()

            if seed_path.exists():
                # If already trained (for a restart)
                model = PPO.load(str(seed_path), env=MaritimeEnv(cost_map, fields=fields), device="cpu")
                print(f"loaded from cache", end=" ")
            else:
                model = train_ppo_seed(cost_map, seed, PPO_TIMESTEPS, seed_path, fields=fields)

            # Evaluation
            ep_results = evaluate_ppo(model, cost_map, N_EVAL_EPISODES, seed, fields=fields)
            costs   = [r["path_cost"]  for r in ep_results]   # path-integrated cost of the route to the goal
            rewards = [r["reward"]     for r in ep_results]
            arrived = [r["arrived"]    for r in ep_results]
            steps_  = [r["steps"]      for r in ep_results]

            ppo_all_costs.extend(costs)
            ppo_all_rewards.extend(rewards)
            ppo_all_arrived.extend(arrived)
            ppo_all_steps.extend(steps_)
            ppo_all_pcost.extend(r["path_cost"] for r in ep_results)
            ppo_all_pnm.extend(r["path_nm"] for r in ep_results)
            ppo_all_seed.extend([seed] * len(ep_results))

            elapsed = time.time() - t0
            print(f"cost={np.mean(costs):.4f} arrived={np.mean(arrived)*100:.0f}% [{elapsed:.0f}s]")

        # ── Save the raw results ──
        raw = {
            "ppo_costs":   ppo_all_costs,
            "ppo_rewards": ppo_all_rewards,
            "ppo_arrived": ppo_all_arrived,
            "ppo_steps":   ppo_all_steps,
            "baseline_costs": baseline_costs,
            "baseline_costs_cells": baseline_costs_cells,
            "ppo_seed":       ppo_all_seed,
            "ppo_path_costs": ppo_all_pcost,
            "ppo_path_nm":    ppo_all_pnm,
            "baseline_path_costs": {k: v[0] for k, v in baseline_path.items()},
            "baseline_path_nm":    {k: v[1] for k, v in baseline_path.items()},
        }
        with open(scen_dir / "raw_results.json", "w") as f:
            json.dump(raw, f, indent=2)

        all_results[scen_name] = raw

        # ── Bootstrap CI ──
        ppo_mean, ppo_lo, ppo_hi = bootstrap_ci(ppo_all_costs)
        print(f"\n  PPO total_cost: {ppo_mean:.4f} 95%CI [{ppo_lo:.4f}, {ppo_hi:.4f}]")

        # ── t-tests ──
        ttests = []
        for bname, bcost in baseline_costs.items():
            tt = ttest_vs_baseline(ppo_all_costs, bcost, bname)
            ttests.append(tt)
            sig = "✅ p<0.05" if tt["significant"] else "❌ ns"
            print(f"  t-test PPO vs {bname}: t={tt['t_stat']}, p={tt['p_value']} {sig}")

        pd.DataFrame(ttests).to_csv(scen_dir / "ttest_results.csv", index=False)

    # ── Save everything together ──
    save_combined_results(all_results, exp_dir)
    return all_results


# ════════════════════════════════════════════════════════════════════
#  8. SUMMARY TABLES
# ════════════════════════════════════════════════════════════════════

def save_combined_results(all_results, exp_dir):
    """Builds the summary table: mean ± std per scenario."""
    rows = []
    for scen_name, data in all_results.items():
        ppo_costs = np.array(data["ppo_costs"])
        ppo_mean, ppo_lo, ppo_hi = bootstrap_ci(ppo_costs)
        ppo_arr = np.mean(data["ppo_arrived"]) * 100

        row = {
            "Scenario": scen_name,
            "Method":   "PPO",
            "Cost Mean": round(ppo_mean, 4),
            "Cost Std":  round(float(np.std(ppo_costs)), 4),
            "Cost CI Lo": round(ppo_lo, 4),
            "Cost CI Hi": round(ppo_hi, 4),
            "Arrival Rate (%)": round(ppo_arr, 1),
        }
        rows.append(row)

        for bname, bcost in data["baseline_costs"].items():
            rows.append({
                "Scenario": scen_name,
                "Method":   bname,
                "Cost Mean": round(float(bcost), 4),
                "Cost Std":  0.0,
                "Cost CI Lo": round(float(bcost), 4),
                "Cost CI Hi": round(float(bcost), 4),
                "Arrival Rate (%)": 100.0,
            })

    df = pd.DataFrame(rows)
    path = exp_dir / "summary_table.csv"
    df.to_csv(path, index=False)
    print(f"\n✅ Summary table → {path}")
    save_pathcost_summary(all_results, exp_dir)
    return df


def save_pathcost_summary(all_results, exp_dir):
    """Path-integrated cost (cost·nm), seed level: the unit of independence is the training seed
    (N = N_SEEDS means of N_EVAL_EPISODES episodes), 95% t-intervals over seeds."""
    from scipy import stats as st
    rows = []
    for scen_name, data in all_results.items():
        df = pd.DataFrame({"seed": data["ppo_seed"], "cost": data["ppo_path_costs"],
                           "nm": data["ppo_path_nm"], "arrived": data["ppo_arrived"]})
        per_seed = df.groupby("seed").mean()
        n = len(per_seed)
        def tci(x):
            m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))
            h = float(st.t.ppf(0.975, len(x) - 1) * se)
            return m, m - h, m + h
        m, lo, hi = tci(per_seed["cost"])
        row = {"Scenario": scen_name, "PPO cost mean": round(m, 4),
               "PPO cost SD (seeds)": round(float(per_seed["cost"].std(ddof=1)), 4),
               "PPO 95% CI lo": round(lo, 4), "PPO 95% CI hi": round(hi, 4),
               "PPO length nm": round(float(per_seed["nm"].mean()), 2),
               "PPO arrival (%)": round(float(df["arrived"].mean()) * 100, 1), "N seeds": n}
        for b in ["Dijkstra", "A*", "Direct"]:
            bc = data["baseline_path_costs"][b]
            row[f"{b} cost"] = round(bc, 4)
            row[f"{b} length nm"] = round(data["baseline_path_nm"][b], 2)
            d_m, d_lo, d_hi = tci(per_seed["cost"] - bc)
            row[f"PPO-{b} diff"] = round(d_m, 4)
            row[f"PPO-{b} 95% CI"] = f"[{d_lo:.4f}, {d_hi:.4f}]"
            row[f"PPO vs {b} (%)"] = round((m - bc) / bc * 100, 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    path = exp_dir / "summary_pathcost.csv"
    out.to_csv(path, index=False)
    print(f"✅ Path-integrated cost summary → {path}")
    print(out[["Scenario", "PPO cost mean", "Dijkstra cost", "Direct cost", "PPO vs Dijkstra (%)",
               "PPO-Dijkstra 95% CI", "PPO arrival (%)"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════
#  9. PLOTS
# ════════════════════════════════════════════════════════════════════

def plot_all(all_results, fig_dir):
    fig_dir = Path(fig_dir)

    # ── 9.1 Mean±Std per scenario (grouped bar) ──
    scen_names = list(all_results.keys())
    methods    = ["PPO", "Dijkstra", "A*", "Direct"]
    x = np.arange(len(scen_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for mi, method in enumerate(methods):
        means, errs = [], []
        for scen in scen_names:
            data = all_results[scen]
            if method == "PPO":
                costs = np.array(data["ppo_costs"])
                means.append(np.mean(costs))
                errs.append(np.std(costs))
            else:
                c = data["baseline_costs"].get(method, np.nan)
                means.append(c)
                errs.append(0.0)

        bars = ax.bar(x + mi * width, means, width,
                      yerr=errs, capsize=4,
                      color=COLORS.get(method, "#95a5a6"),
                      label=method, alpha=0.88,
                      error_kw=dict(elinewidth=1.5, ecolor="black"))

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scen_names, fontsize=9)
    ax.set_ylabel("Total Route Cost", fontsize=10)
    ax.set_title("Algorithm Comparison Across Scenarios\n(PPO: mean ± std over "
                 f"{N_SEEDS} seeds × {N_EVAL_EPISODES} episodes)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_exp_scenario_comparison.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig_exp_scenario_comparison.png")

    # ── 9.2 Violin plot — distribution of the PPO cost ──
    fig, axes = plt.subplots(1, len(scen_names),
                              figsize=(4 * len(scen_names), 4.5), sharey=False)
    if len(scen_names) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scen_names):
        costs = np.array(all_results[scen]["ppo_costs"])
        vp = ax.violinplot([costs], showmeans=True, showmedians=True)
        vp["bodies"][0].set_facecolor(SCENARIO_COLORS.get(scen, "#3498db"))
        vp["bodies"][0].set_alpha(0.7)
        mean, lo, hi = bootstrap_ci(costs)
        ax.hlines([lo, hi], 0.7, 1.3, colors="red",
                  linestyles="--", linewidth=1.5, label="95% CI")
        ax.set_title(scen, fontsize=9, fontweight="bold")
        ax.set_ylabel("Total Cost" if ax == axes[0] else "")
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"PPO Cost Distribution per Scenario\n"
                 f"({N_SEEDS} seeds × {N_EVAL_EPISODES} eps, dashed = 95% CI)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_exp_violin.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig_exp_violin.png")

    # ── 9.3 Heatmap p-values (PPO vs baselines × scenarios) ──
    baselines = ["Dijkstra", "A*", "Direct"]
    pmat = np.zeros((len(scen_names), len(baselines)))
    for si, scen in enumerate(scen_names):
        ppo_costs = np.array(all_results[scen]["ppo_costs"])
        for bi, bname in enumerate(baselines):
            bcost = all_results[scen]["baseline_costs"][bname]
            _, p = stats.ttest_1samp(ppo_costs, bcost)
            pmat[si, bi] = p

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=0.1,
                   aspect="auto")
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, fontsize=9)
    ax.set_yticks(range(len(scen_names)))
    ax.set_yticklabels(scen_names, fontsize=9)
    for si in range(len(scen_names)):
        for bi in range(len(baselines)):
            p = pmat[si, bi]
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            ax.text(bi, si, f"p={p:.3f}\n{sig}",
                    ha="center", va="center", fontsize=8,
                    color="white" if p < 0.05 else "black")
    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_title("t-test p-values: PPO vs Baselines\n(* p<0.05, ** p<0.01)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_exp_pvalues.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig_exp_pvalues.png")

    # ── 9.4 Arrival rate × scenario ──
    fig, ax = plt.subplots(figsize=(8, 4))
    arr_rates = [np.mean(all_results[s]["ppo_arrived"]) * 100
                 for s in scen_names]
    bars = ax.bar(scen_names, arr_rates,
                  color=[SCENARIO_COLORS.get(s, "#3498db") for s in scen_names],
                  edgecolor="white", linewidth=1.2, alpha=0.88)
    for bar, val in zip(bars, arr_rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Arrival Rate (%)", fontsize=10)
    ax.set_title("PPO Arrival Rate by Scenario",
                 fontsize=10, fontweight="bold")
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_exp_arrival.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig_exp_arrival.png")

    # ── 9.5 Cost reduction % vs Direct ──
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(scen_names))
    width = 0.25
    algo_list = ["PPO", "Dijkstra", "A*"]
    for mi, method in enumerate(algo_list):
        reductions = []
        for scen in scen_names:
            direct_c = all_results[scen]["baseline_costs"]["Direct"]
            if method == "PPO":
                c = np.mean(all_results[scen]["ppo_costs"])
            else:
                c = all_results[scen]["baseline_costs"].get(method, direct_c)
            reductions.append((direct_c - c) / direct_c * 100)
        ax.bar(x + mi * width, reductions, width,
               color=COLORS.get(method, "#95a5a6"),
               label=method, alpha=0.88, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scen_names, fontsize=9)
    ax.set_ylabel("Cost Reduction vs Direct (%)", fontsize=10)
    ax.set_title("Cost Reduction vs Direct Route — All Scenarios",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_exp_cost_reduction.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig_exp_cost_reduction.png")

    print(f"\n✅ All figures → {fig_dir}")


# ════════════════════════════════════════════════════════════════════
#  10. MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  OVERNIGHT EXPERIMENT RUNNER")
    print(f"  Seeds: {N_SEEDS}  |  Timesteps/seed: {PPO_TIMESTEPS:,}")
    print(f"  Scenarios: {list(SCENARIOS.keys())}")
    print(f"  Eval episodes/seed: {N_EVAL_EPISODES}")
    print("=" * 60)

    # Check the dependencies
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 is not installed. Installing...")
        os.system(f"{sys.executable} -m pip install stable-baselines3")
        from stable_baselines3 import PPO

    t_start = time.time()

    all_results = run_all_experiments()
    plot_all(all_results, FIGURES_DIR)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for scen, data in all_results.items():
        ppo_mean = np.mean(data["ppo_costs"])
        ppo_std  = np.std(data["ppo_costs"])
        arr_rate = np.mean(data["ppo_arrived"]) * 100
        direct_c = data["baseline_costs"]["Direct"]
        red_pct  = (direct_c - ppo_mean) / direct_c * 100
        print(f"  {scen:20s}: PPO cost {ppo_mean:.4f}±{ppo_std:.4f} "
              f"| arrival {arr_rate:.0f}% | vs Direct {red_pct:+.1f}%")

    elapsed = (time.time() - t_start) / 3600
    print(f"\n  Total time: {elapsed:.1f} h")
    print(f"  Results: {EXP_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
