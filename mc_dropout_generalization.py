"""
mc_dropout_generalization.py
-----------------------------
Part 1: Monte Carlo Dropout — uncertainty-aware LSTM predictions
Part 2: Generalization — Houston-trained model tested on Chesapeake & Puget Sound

Usage:
  python mc_dropout_generalization.py

What it does:
  1. Loads the trained LSTM model (results/lstm_model_temporal.pt of lstm_predictor_temporal_split.py)
  2. Keeps Dropout active at inference (MC Dropout)
  3. Makes T=30 forward passes → mean + std of the prediction
  4. Computes the uncertainty of the wave height prediction (sigma_forecast)
  5. Tests the model on Chesapeake Bay and Puget Sound data
  6. Draws the uncertainty and generalization figures

Output:
  results/figures/ieee/fig_mc_dropout.png
  results/figures/ieee/fig_generalization.png
  results/figures/ieee/fig_uncertainty.png
"""

import os, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR     = os.path.join(ROOT, "results", "figures", "ieee")

# AIS merged folders of each region
REGIONS = {
    "Houston":    os.path.join(ROOT, "ais_merged"),
    "Chesapeake": os.path.join(ROOT, "ais_merged_chesapeake"),
    "PugetSound": os.path.join(ROOT, "ais_merged_pugetsound"),
}

# LSTM parameters (must match lstm_predictor.py)
SEQ_LEN    = 12
PRED_LEN   = 6
HIDDEN     = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 256
MC_SAMPLES = 30   # number of MC forward passes

INPUT_FEATURES  = ["wind_u", "wind_v", "wind_speed", "wave_hs", "SOG", "COG", "LAT", "LON"]
TARGET_FEATURES = ["wind_u", "wind_v", "wave_hs"]

TARGET_LABELS = {
    "wind_u":  "Wind U-component (m/s)",
    "wind_v":  "Wind V-component (m/s)",
    "wave_hs": "Significant Wave Height (m)",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "figure.dpi": 300,
})


# ════════════════════════════════════════════════════════════════════
#  MODEL WITH MC DROPOUT
# ════════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_targets,
                 hidden=HIDDEN, num_layers=NUM_LAYERS,
                 pred_len=PRED_LEN, dropout=DROPOUT):
        super().__init__()
        self.pred_len  = pred_len
        self.n_targets = n_targets
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),   # ← this Dropout is active in .train()
            nn.Linear(hidden // 2, pred_len * n_targets),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        pred   = self.fc(last)
        return pred.view(-1, self.pred_len, self.n_targets)


def enable_mc_dropout(model):
    """Keeps Dropout active at inference (MC Dropout trick)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()   # only Dropout in train mode, everything else in eval
    return model


# ════════════════════════════════════════════════════════════════════
#  LOADING THE DATA OF A REGION
# ════════════════════════════════════════════════════════════════════

def load_region_data(region_name, ais_dir, max_files=20, name_filter=""):
    """Loads the data of a region and returns X, y (sequences)."""
    files = [f for f in sorted(Path(ais_dir).glob("*_era5.parquet")) if name_filter in f.name][:max_files]
    if not files:
        print(f"  ⚠️  No files in {ais_dir}")
        return None, None, None

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)

    # Time
    time_col = next((c for c in ["BaseDateTime","timestamp","time"]
                     if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.rename(columns={time_col: "timestamp"}).sort_values("timestamp")

    # Cleaning
    key_cols = [c for c in TARGET_FEATURES if c in df.columns]
    df = df.dropna(subset=key_cols)
    if "SOG" in df.columns:
        df = df[df["SOG"].between(0, 30)]

    in_cols  = [c for c in INPUT_FEATURES  if c in df.columns]
    out_cols = [c for c in TARGET_FEATURES if c in df.columns]

    if len(in_cols) < 4:
        print(f"  ⚠️  Not enough features for {region_name}: {in_cols}")
        return None, None, None

    X_list, y_list = [], []
    group_col = "MMSI" if "MMSI" in df.columns else None
    groups = df.groupby(group_col) if group_col else [(None, df)]

    for _, grp in groups:
        grp = grp.reset_index(drop=True)
        arr_in  = grp[in_cols].values.astype(np.float32)
        arr_out = grp[out_cols].values.astype(np.float32)
        total = SEQ_LEN + PRED_LEN
        if len(grp) < total:
            continue
        for i in range(min(len(grp) - total + 1, 200)):  # at most 200 per vessel
            X_list.append(arr_in[i:i+SEQ_LEN])
            y_list.append(arr_out[i+SEQ_LEN:i+total])

    if not X_list:
        print(f"  ⚠️  No sequences for {region_name}")
        return None, None, None

    X = np.stack(X_list[:5000])  # at most 5000 sequences
    y = np.stack(y_list[:5000])
    print(f"  {region_name}: {len(X):,} sequences, "
          f"{len(in_cols)} features")
    return X, y, out_cols


# ════════════════════════════════════════════════════════════════════
#  MC DROPOUT INFERENCE
# ════════════════════════════════════════════════════════════════════

def mc_predict(model, X_norm, device, T=MC_SAMPLES):
    """
    T forward passes with Dropout active.
    Returns the mean and std of the predictions.
    """
    all_preds = []
    with torch.no_grad():
        for _ in range(T):
            preds_t = []
            for i in range(0, len(X_norm), BATCH_SIZE):
                xb = torch.tensor(X_norm[i:i+BATCH_SIZE]).to(device)
                preds_t.append(model(xb).cpu().numpy())
            all_preds.append(np.concatenate(preds_t))

    all_preds = np.stack(all_preds)  # (T, N, pred_len, n_targets)
    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)
    return mean_pred, std_pred


def evaluate_region(model, scaler_x, scaler_y, X, y, out_cols, device):
    """Evaluates the model on a region with MC Dropout."""
    B, T, F = X.shape
    # Normalise with the Houston scaler (zero-shot generalization)
    try:
        X_norm = scaler_x.transform(X.reshape(-1, F)).reshape(B, T, F)
    except Exception as e:
        print(f"    Normalisation error: {e}")
        return None

    mean_pred_norm, std_pred_norm = mc_predict(model, X_norm, device)

    # Inverse normalisation
    _, T_out, C = mean_pred_norm.shape
    mean_pred = scaler_y.inverse_transform(
        mean_pred_norm.reshape(-1, C)).reshape(-1, T_out, C)
    # std is rescaled with the scaler scale_
    std_pred = std_pred_norm * scaler_y.scale_[np.newaxis, np.newaxis, :]

    # Metrics
    mae  = np.mean(np.abs(mean_pred - y), axis=(0, 1))   # (n_targets,)
    rmse = np.sqrt(np.mean((mean_pred - y)**2, axis=(0, 1)))
    mean_uncertainty = std_pred.mean(axis=(0, 1))

    return {
        "mae":         mae,
        "rmse":        rmse,
        "uncertainty": mean_uncertainty,
        "mean_pred":   mean_pred,
        "std_pred":    std_pred,
        "y_true":      y,
        "out_cols":    out_cols,
    }


# ════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════

def plot_mc_dropout(houston_res, fig_dir):
    """
    Fig A: prediction with uncertainty bands for 4 examples.
    Blue line = ground truth, red = mean prediction,
    pink band = ±2σ (95% prediction interval).
    """
    fig_dir = Path(fig_dir)
    preds   = houston_res["mean_pred"]
    stds    = houston_res["std_pred"]
    trues   = houston_res["y_true"]
    out_cols = houston_res["out_cols"]

    n_targets = len(out_cols)
    n_samples = 4
    hours     = np.arange(1, PRED_LEN + 1)

    # Choose dynamic examples
    ranges = [sum(trues[i, :, c].max() - trues[i, :, c].min()
                  for c in range(n_targets))
              for i in range(len(trues))]
    top_idx = np.argsort(ranges)[-200:]
    step    = max(1, len(top_idx) // (n_samples * 5))
    sel     = top_idx[::step][:n_samples*3]
    idx     = np.linspace(0, len(sel)-1, n_samples, dtype=int)
    sample_idx = sel[idx]

    fig, axes = plt.subplots(n_targets, n_samples,
                              figsize=(7.16, 2.5*n_targets))
    if n_targets == 1:
        axes = axes[np.newaxis, :]

    for row, (col_idx, name) in enumerate(zip(range(n_targets), out_cols)):
        label = TARGET_LABELS.get(name, name)
        for col, si in enumerate(sample_idx):
            ax = axes[row, col]
            y_true = trues[si, :, col_idx]
            y_mean = preds[si, :, col_idx]
            y_std  = stds[si, :, col_idx]

            ax.plot(hours, y_true, "o-", color="#1a6faf",
                    lw=1.8, ms=4, label="Ground truth", zorder=3)
            ax.plot(hours, y_mean, "s--", color="#c0392b",
                    lw=1.8, ms=4, label="MC mean", zorder=3)
            ax.fill_between(hours,
                            y_mean - 2*y_std,
                            y_mean + 2*y_std,
                            alpha=0.25, color="#e74c3c",
                            label="±2σ (95% PI)")

            mae = np.mean(np.abs(y_true - y_mean))
            ax.text(0.97, 0.05, f"MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            ax.set_xlabel("Step (AIS reports)", fontsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"Sample {col+1}", fontsize=7, pad=2)
            ax.set_xticks(hours)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=6)
            if row == 0 and col == n_samples-1:
                ax.legend(fontsize=6, loc="upper right",
                          framealpha=0.85)

    fig.suptitle(
        f"MC Dropout Uncertainty Estimation (T={MC_SAMPLES} samples)\n"
        "Houston Ship Channel — Ground Truth vs. Mean Prediction ± 2σ",
        fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in [".png", ".pdf"]:
        fig.savefig(fig_dir / f"fig_mc_dropout{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✅ fig_mc_dropout.png")


def plot_generalization(results_by_region, fig_dir):
    """
    Fig B: MAE comparison by region — grouped bar.
    Shows the zero-shot generalization Houston → other regions.
    """
    fig_dir   = Path(fig_dir)
    regions   = [r for r in results_by_region if results_by_region[r] is not None]
    out_cols  = results_by_region[regions[0]]["out_cols"]
    n_targets = len(out_cols)

    colors = {
        "Houston":    "#1a6faf",
        "Chesapeake": "#e05c2a",
        "PugetSound": "#27ae60",
    }
    labels_nice = {
        "Houston":    "Houston\n(June test)",
        "Chesapeake": "Chesapeake\n(zero-shot)",
        "PugetSound": "Puget Sound\n(zero-shot)",
    }

    fig, axes = plt.subplots(1, n_targets, figsize=(7.16, 2.8))
    if n_targets == 1:
        axes = [axes]

    for ax, (col_idx, name) in zip(axes, enumerate(out_cols)):
        label = TARGET_LABELS.get(name, name)
        x     = np.arange(len(regions))
        maes  = [results_by_region[r]["mae"][col_idx] for r in regions]
        rmses = [results_by_region[r]["rmse"][col_idx] for r in regions]
        uncerts = [results_by_region[r]["uncertainty"][col_idx] for r in regions]

        width = 0.28
        bars1 = ax.bar(x - width, maes,  width, label="MAE",
                       color=[colors.get(r,"#95a5a6") for r in regions],
                       alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x,          rmses, width, label="RMSE",
                       color=[colors.get(r,"#95a5a6") for r in regions],
                       alpha=0.5,  edgecolor="white", hatch="///")
        bars3 = ax.bar(x + width,  uncerts, width, label="Mean σ",
                       color=[colors.get(r,"#95a5a6") for r in regions],
                       alpha=0.35, edgecolor="white", hatch="xxx")

        for bar, val in zip(bars1, maes):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels([labels_nice.get(r, r) for r in regions], fontsize=7)
        ax.set_ylabel(f"Error — {label}", fontsize=7)
        ax.set_title(f"{name}", fontsize=8)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.85)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Cross-Region Generalization: Houston-Trained LSTM\n"
        "Zero-Shot Evaluation on Chesapeake Bay and Puget Sound",
        fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in [".png", ".pdf"]:
        fig.savefig(fig_dir / f"fig_generalization{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✅ fig_generalization.png")


def plot_uncertainty_on_map(houston_res, fig_dir):
    """
    Fig C: uncertainty — mean σ of the wave_hs prediction per sample.
    Shows the distribution of the uncertainty.
    """
    fig_dir  = Path(fig_dir)
    stds     = houston_res["std_pred"]  # (N, pred_len, n_targets)
    out_cols = houston_res["out_cols"]

    # LAT/LON would be needed for a map - not available in the prediction data
    # so the figure is drawn from the per-sample statistics
    wave_idx = out_cols.index("wave_hs") if "wave_hs" in out_cols else 0
    mean_sigma = stds[:, :, wave_idx].mean(axis=1)  # (N,) — mean uncertainty

    # Simple scatter over the sample index (no lat/lon in the prediction data)
    # Take the first 1000 examples
    n = min(1000, len(mean_sigma))
    sigma_vals = mean_sigma[:n]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    sc = ax.scatter(np.arange(n), sigma_vals, c=sigma_vals,
                    cmap="YlOrRd", s=6, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Mean σ (m)")
    ax.set_xlabel("Sample index", fontsize=8)
    ax.set_ylabel("Prediction uncertainty σ (m)", fontsize=8)
    ax.set_title("Wave Height Forecast Uncertainty\n"
                 f"(MC Dropout, T={MC_SAMPLES})", fontsize=8)
    ax.axhline(sigma_vals.mean(), color="#1a6faf", linestyle="--",
               lw=1.5, label=f"Mean σ = {sigma_vals.mean():.4f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(fig_dir / f"fig_uncertainty{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✅ fig_uncertainty.png")


def print_generalization_table(results_by_region):
    """Prints the table for the paper."""
    print("\n" + "="*65)
    print("  GENERALIZATION TABLE (for the paper)")
    print("="*65)
    print(f"  {'Region':<15} {'Variable':<12} {'MAE':>8} {'RMSE':>8} {'Mean σ':>8}")
    print(f"  {'─'*55}")

    for region, res in results_by_region.items():
        if res is None:
            print(f"  {region:<15} — no data")
            continue
        for ci, name in enumerate(res["out_cols"]):
            tag = "(June test)" if region == "Houston" else "(zero-shot)"
            print(f"  {region+' '+tag:<22} {name:<12} "
                  f"{res['mae'][ci]:>8.4f} "
                  f"{res['rmse'][ci]:>8.4f} "
                  f"{res['uncertainty'][ci]:>8.4f}")
        print()


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"  MC DROPOUT + GENERALIZATION")
    print(f"  MC samples: {MC_SAMPLES}")
    print("="*60)

    # Load the scalers and the model
    print("\nLoading the model and the scalers...")
    scaler_x = joblib.load(Path(RESULTS_DIR) / "scaler_x_temporal.pkl")
    scaler_y = joblib.load(Path(RESULTS_DIR) / "scaler_y_temporal.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load the Houston data to get n_features
    print("\nLoading the Houston data...")
    X_h, y_h, out_cols = load_region_data(
        "Houston", REGIONS["Houston"], max_files=15, name_filter="_2024_06_")   # June = test month
    if X_h is None:
        print("❌ Could not load the Houston data")
        return

    n_features = X_h.shape[2]
    n_targets  = y_h.shape[2]
    print(f"  n_features={n_features}, n_targets={n_targets}")

    # Initialise the model
    model = LSTMPredictor(n_features, n_targets).to(device)
    model.load_state_dict(torch.load(
        Path(RESULTS_DIR) / "lstm_model_temporal.pt", map_location=device))

    # Enable MC Dropout
    model = enable_mc_dropout(model)
    print(f"  MC Dropout enabled (T={MC_SAMPLES} passes)")

    # Normalise the Houston data
    B, T, F = X_h.shape
    X_h_norm = scaler_x.transform(X_h.reshape(-1, F)).reshape(B, T, F)

    # ── Evaluate all regions ──
    results_by_region = {}
    for region_name, ais_dir in REGIONS.items():
        print(f"\nEvaluating {region_name}...")
        if region_name == "Houston":
            X, y = X_h, y_h
            X_norm = X_h_norm
        else:
            X, y, _ = load_region_data(region_name, ais_dir, max_files=10)
            if X is None:
                results_by_region[region_name] = None
                continue
            try:
                Bx, Tx, Fx = X.shape
                # Padding if there are fewer features
                if Fx < F:
                    pad = np.zeros((Bx, Tx, F - Fx), dtype=np.float32)
                    X   = np.concatenate([X, pad], axis=2)
                    Fx  = F
                X_norm = scaler_x.transform(
                    X.reshape(-1, Fx)).reshape(Bx, Tx, Fx)
            except Exception as e:
                print(f"  ⚠️  Normalisation error {region_name}: {e}")
                results_by_region[region_name] = None
                continue

        res = evaluate_region(model, scaler_x, scaler_y,
                              X, y, out_cols, device)
        results_by_region[region_name] = res

        if res:
            for ci, name in enumerate(out_cols):
                print(f"  {name}: MAE={res['mae'][ci]:.4f} "
                      f"RMSE={res['rmse'][ci]:.4f} "
                      f"σ={res['uncertainty'][ci]:.4f}")

    # ── Figures ──
    print("\nDrawing the figures...")
    if results_by_region.get("Houston"):
        plot_mc_dropout(results_by_region["Houston"], FIG_DIR)
        plot_uncertainty_on_map(results_by_region["Houston"], FIG_DIR)

    valid_regions = {k: v for k, v in results_by_region.items()
                     if v is not None}
    if len(valid_regions) > 1:
        plot_generalization(valid_regions, FIG_DIR)

    print_generalization_table(results_by_region)

    print(f"\n✅ Done! Figures → {FIG_DIR}")
    print("  fig_mc_dropout.png  — uncertainty bands")
    print("  fig_uncertainty.png — σ distribution")
    print("  fig_generalization.png — cross-region MAE")


if __name__ == "__main__":
    main()
