"""
plot_predictions.py
----------------------
Publication figures of the LSTM predictions (model of lstm_predictor_temporal_split.py,
June test data).
Finds dynamic examples (where the values really change)
and draws publication-quality figures.

Usage:
  python plot_predictions.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd

# ─── paths ───
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
AIS_MERGED  = os.path.join(ROOT, "ais_merged")

# ─── parameters (must match lstm_predictor.py) ───
SEQ_LEN    = 12
PRED_LEN   = 6
HIDDEN     = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 256

INPUT_FEATURES  = ["wind_u", "wind_v", "wind_speed", "wave_hs", "SOG", "COG", "LAT", "LON"]
TARGET_FEATURES = ["wind_u", "wind_v", "wave_hs"]

TARGET_LABELS = {
    "wind_u":  "Wind U-component (m/s)",
    "wind_v":  "Wind V-component (m/s)",
    "wave_hs": "Significant Wave Height (m)",
}

# IEEE colours
COLOR_TRUE = "#1f77b4"   # blue
COLOR_PRED = "#d62728"   # red
COLOR_FILL = "#aec7e8"   # light blue for the CI


# ═══════════════════════════════════════════════════════════════
#  Model (copy from lstm_predictor.py)
# ═══════════════════════════════════════════════════════════════

import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_targets, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, pred_len=PRED_LEN, dropout=DROPOUT):
        super().__init__()
        self.pred_len  = pred_len
        self.n_targets = n_targets
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, pred_len * n_targets),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred.view(-1, self.pred_len, self.n_targets)


# ═══════════════════════════════════════════════════════════════
#  Loading the data and the model
# ═══════════════════════════════════════════════════════════════

def load_everything():
    print("Loading data...")
    files = sorted(Path(AIS_MERGED).glob("*_era5.parquet"))
    files = [f for f in files if "_2024_06_" in f.name]   # June = test month of the temporal split
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Time
    time_col = next((c for c in ["BaseDateTime","timestamp","time"] if c in df.columns), None)
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

    # Sequences
    X_list, y_list, ts_list = [], [], []
    for _, grp in df.groupby("MMSI"):
        grp = grp.reset_index(drop=True)
        arr_in  = grp[in_cols].values.astype(np.float32)
        arr_out = grp[out_cols].values.astype(np.float32)
        total = SEQ_LEN + PRED_LEN
        if len(grp) < total:
            continue
        for i in range(len(grp) - total + 1):
            X_list.append(arr_in[i:i+SEQ_LEN])
            y_list.append(arr_out[i+SEQ_LEN:i+total])

    X = np.stack(X_list)
    y = np.stack(y_list)

    # Normalisation
    scaler_x = joblib.load(Path(RESULTS_DIR) / "scaler_x_temporal.pkl")
    scaler_y = joblib.load(Path(RESULTS_DIR) / "scaler_y_temporal.pkl")

    B, T, F = X.shape
    X_norm = scaler_x.transform(X.reshape(-1, F)).reshape(B, T, F)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(len(in_cols), len(out_cols)).to(device)
    model.load_state_dict(torch.load(
        Path(RESULTS_DIR) / "lstm_model_temporal.pt", map_location=device))
    model.eval()

    # Predictions in batches
    preds_norm = []
    with torch.no_grad():
        for i in range(0, len(X_norm), BATCH_SIZE):
            xb = torch.tensor(X_norm[i:i+BATCH_SIZE]).to(device)
            preds_norm.append(model(xb).cpu().numpy())
    preds_norm = np.concatenate(preds_norm)

    # Inverse normalisation
    _, T_out, C = preds_norm.shape
    preds = scaler_y.inverse_transform(preds_norm.reshape(-1, C)).reshape(-1, T_out, C)
    trues = y  # already in the original units

    print(f"Predictions: {len(preds):,}")
    return preds, trues, out_cols


# ═══════════════════════════════════════════════════════════════
#  Finding dynamic examples
# ═══════════════════════════════════════════════════════════════

def find_dynamic_samples(trues, n=4, min_range_pct=70):
    """
    Finds samples where the true values really change.
    min_range_pct — minimum percentile of the range of values.
    """
    # For every sample compute the total range over all targets
    ranges = []
    for i in range(len(trues)):
        r = sum(trues[i, :, c].max() - trues[i, :, c].min()
                for c in range(trues.shape[2]))
        ranges.append(r)
    ranges = np.array(ranges)

    # Keep only those above the min_range_pct percentile
    threshold = np.percentile(ranges, min_range_pct)
    candidates = np.where(ranges >= threshold)[0]

    # Take evenly spaced candidates
    step = max(1, len(candidates) // (n * 10))
    selected = candidates[::step][:n*5]
    # Add variety - take them from different parts
    idx = np.linspace(0, len(selected)-1, n, dtype=int)
    return selected[idx]


# ═══════════════════════════════════════════════════════════════
#  Main figure — publication quality
# ═══════════════════════════════════════════════════════════════

def plot_publication(preds, trues, out_cols, figures_dir):
    """
    3 rows (one per target) × 4 columns (4 different examples).
    IEEE two-column style.
    """
    n_targets = len(out_cols)
    n_samples = 4

    dynamic_idx = find_dynamic_samples(trues, n=n_samples)
    print(f"Selected dynamic examples: {dynamic_idx}")

    hours = np.arange(1, PRED_LEN + 1)

    fig = plt.figure(figsize=(14, 3.5 * n_targets))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(n_targets, n_samples, figure=fig,
                           hspace=0.45, wspace=0.35)

    for row, (col_idx, name) in enumerate(zip(range(n_targets), out_cols)):
        label = TARGET_LABELS.get(name, name)

        for col, sample_i in enumerate(dynamic_idx):
            ax = fig.add_subplot(gs[row, col])

            y_true = trues[sample_i, :, col_idx]
            y_pred = preds[sample_i, :, col_idx]

            # Main lines
            ax.plot(hours, y_true, "o-",
                    color=COLOR_TRUE, linewidth=1.8, markersize=5,
                    label="Ground truth", zorder=3)
            ax.plot(hours, y_pred, "s--",
                    color=COLOR_PRED, linewidth=1.8, markersize=5,
                    label="LSTM prediction", zorder=3)

            # Fill between the curves
            ax.fill_between(hours, y_true, y_pred,
                            alpha=0.15, color=COLOR_PRED)

            # MAE of this example
            mae = np.mean(np.abs(y_true - y_pred))
            ax.text(0.97, 0.05, f"MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            ax.set_xlabel("Forecast step (AIS reports, ~3 min each)", fontsize=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

            ax.set_title(f"Sample {col+1}", fontsize=8, pad=3)
            ax.set_xticks(hours)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)

            if row == 0 and col == n_samples - 1:
                ax.legend(fontsize=7, loc="upper right",
                          framealpha=0.8, edgecolor="gray")

    fig.suptitle(
        "LSTM Multi-step Forecasting: Ground Truth vs Predictions\n"
        "Houston Ship Channel, Jan–Jun 2024",
        fontsize=11, fontweight="bold", y=1.01
    )

    path = Path(figures_dir) / "fig_predictions_v2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {path}")

    # Additional figure - error by horizon
    plot_error_by_horizon(preds, trues, out_cols, figures_dir)


def plot_error_by_horizon(preds, trues, out_cols, figures_dir):
    """
    MAE by prediction horizon - the standard figure of ML papers.
    """
    hours = np.arange(1, PRED_LEN + 1)
    n_targets = len(out_cols)

    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 3.5))
    if n_targets == 1:
        axes = [axes]

    for ax, (col_idx, name) in zip(axes, enumerate(out_cols)):
        label = TARGET_LABELS.get(name, name)

        # MAE at every horizon step
        mae_per_h = [
            np.mean(np.abs(preds[:, h, col_idx] - trues[:, h, col_idx]))
            for h in range(PRED_LEN)
        ]
        rmse_per_h = [
            np.sqrt(np.mean((preds[:, h, col_idx] - trues[:, h, col_idx])**2))
            for h in range(PRED_LEN)
        ]

        ax.plot(hours, mae_per_h,  "o-", color=COLOR_TRUE,
                linewidth=2, markersize=6, label="MAE")
        ax.plot(hours, rmse_per_h, "s--", color=COLOR_PRED,
                linewidth=2, markersize=6, label="RMSE")

        ax.fill_between(hours, mae_per_h, rmse_per_h,
                        alpha=0.15, color=COLOR_PRED)

        ax.set_xlabel("Forecast step (AIS reports, ~3 min each)", fontsize=9)
        ax.set_ylabel(f"Error — {label}", fontsize=9)
        ax.set_title(f"{name} forecast error", fontsize=9)
        ax.set_xticks(hours)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("LSTM Forecast Error by Horizon", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()

    path = Path(figures_dir) / "fig_error_by_horizon.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    preds, trues, out_cols = load_everything()
    plot_publication(preds, trues, out_cols, FIGURES_DIR)

    print("\n✅ All figures are in:", FIGURES_DIR)
