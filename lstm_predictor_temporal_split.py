"""
lstm_predictor_temporal_split.py  —  Module 1 of the paper  (temporal split version)
-----------------------------------------------------------------
LSTM predictor of the meteorological fields (wind, waves) for the Houston Ship Channel.

Key difference from lstm_predictor.py:
  The train/val/test split is made by TIME at the DataFrame level
  BEFORE the sliding windows are cut. This removes the data leakage
  of the random split, where neighbouring overlapping sequences
  (overlap = 17/18 steps) could end up in both train and test
  when the split was made by sequence index.

  Split:
    Train : January – April  (≈67%)
    Val   : May              (≈17%)
    Test  : June             (≈17%)  ← never seen during training

Input:  AIS+ERA5 merged parquet files from ais_merged/
Output: trained model    results/lstm_model_temporal.pt
        metrics          results/metrics_lstm_temporal.csv
        figures          results/figures/fig_loss_curve_temporal.png
                         results/figures/fig_predictions_temporal.png

Usage:
  python lstm_predictor_temporal_split.py            # training
  python lstm_predictor_temporal_split.py --check    # quick data check
  python lstm_predictor_temporal_split.py --epochs 5 # quick test
"""

import os
import gc
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────── configuration ───────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository folder; all data and results live here

AIS_MERGED_DIR = os.path.join(ROOT, "ais_merged")
RESULTS_DIR    = os.path.join(ROOT, "results")
FIGURES_DIR    = os.path.join(ROOT, "results", "figures")

# LSTM hyperparameters (same as lstm_predictor.py)
SEQ_LEN    = 12   # input sequence (12 consecutive AIS reports of a vessel)
PRED_LEN   = 6    # prediction horizon (the next 6 AIS reports)
HIDDEN     = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 256
STRIDE     = 6    # windows start every 6 reports (targets do not overlap); stride 1 gives ~8 M
                  # strongly overlapping training windows, which do not fit into 32 GB RAM
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 7

# Temporal split (whole months)
TRAIN_MONTHS = [1, 2, 3, 4]   # January–April
VAL_MONTHS   = [5]             # May
TEST_MONTHS  = [6]             # June  ← never seen during training

INPUT_FEATURES  = ["wind_u", "wind_v", "wind_speed", "wave_hs", "SOG", "COG", "LAT", "LON"]
TARGET_FEATURES = ["wind_u", "wind_v", "wave_hs"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. Data loading and preparation
# ═══════════════════════════════════════════════════════════════════

def load_data(merged_dir: str) -> pd.DataFrame:
    """Loads all *_era5.parquet files and concatenates them."""
    files = sorted(Path(merged_dir).glob("*_era5.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_era5.parquet files in {merged_dir}")

    log.info(f"Loading {len(files)} files from {merged_dir}...")
    # only the columns that are used (the full files have 23 columns and ~18 M rows)
    wanted = ["MMSI", "BaseDateTime", "timestamp", "time"] + INPUT_FEATURES
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df[[c for c in wanted if c in df.columns]])
        except Exception as e:
            log.warning(f"Error reading {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and prepares the data frame."""
    time_col = next((c for c in ["BaseDateTime", "timestamp", "time"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.rename(columns={time_col: "timestamp"})
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        raise ValueError("Time column not found (BaseDateTime / timestamp / time)")

    missing = [c for c in INPUT_FEATURES if c not in df.columns]
    if missing:
        log.warning(f"Missing columns: {missing} - filled with zeros")
        for c in missing:
            df[c] = 0.0

    # Rows without ERA5 target values are dropped, as in lstm_predictor.py
    # (wave_hs is NaN where the ERA5 wave grid has no sea cell).
    # An earlier version filled them with a global ffill over time, which copied
    # values between different vessels.
    key_cols = [c for c in TARGET_FEATURES if c in df.columns]
    df = df.dropna(subset=key_cols)

    if "SOG" in df.columns:
        df = df[df["SOG"].between(0, 30)]

    log.info(f"After cleaning: {len(df):,} rows")
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    """
    Splits the DataFrame by date BEFORE the sliding windows are cut.
    Split (data January–June 2024):
      Train : 2024-01-01 – 2024-04-30
      Val   : 2024-05-01 – 2024-05-31
      Test  : 2024-06-01 – 2024-06-30
    """
    # Find the time column - BaseDateTime in the raw data
    time_col = next((c for c in ["BaseDateTime", "timestamp", "time"] if c in df.columns), None)
    if time_col is None:
        raise ValueError("Time column not found in the DataFrame")

    # Parse as UTC, then drop tz for the comparison
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)

    log.info(f"  timestamp dtype for the split: {ts.dtype}")
    log.info(f"  Range: {ts.min()} → {ts.max()}")
    log.info(f"  Unique months: {sorted(ts.dt.month.unique().tolist())}")

    val_start  = pd.Timestamp("2024-05-01")
    test_start = pd.Timestamp("2024-06-01")
    test_end   = pd.Timestamp("2024-07-01")

    mask_train = ts < val_start
    mask_val   = (ts >= val_start)  & (ts < test_start)
    mask_test  = (ts >= test_start) & (ts < test_end)

    df_train = df[mask_train.values].reset_index(drop=True)
    df_val   = df[mask_val.values].reset_index(drop=True)
    df_test  = df[mask_test.values].reset_index(drop=True)

    log.info(f"Temporal split:")
    log.info(f"  Train (Jan–Apr): {len(df_train):,} rows")
    log.info(f"  Val   (May):     {len(df_val):,} rows")
    log.info(f"  Test  (June):    {len(df_test):,} rows")

    if len(df_val) == 0 or len(df_test) == 0:
        raise ValueError(
            "Val or Test is empty after the temporal split. "
            "Check that the data cover May and June 2024."
        )

    return df_train, df_val, df_test


def make_sequences(df: pd.DataFrame, seq_len: int, pred_len: int):
    """
    Cuts the DataFrame into sequences per vessel (MMSI).
    Returns X (N, seq_len, n_features), y (N, pred_len, n_targets).
    """
    X_list, y_list = [], []

    group_col = "MMSI" if "MMSI" in df.columns else None
    groups = df.groupby(group_col) if group_col else [(None, df)]

    in_cols  = [c for c in INPUT_FEATURES  if c in df.columns]
    out_cols = [c for c in TARGET_FEATURES if c in df.columns]

    total = seq_len + pred_len
    for _, grp in groups:
        grp = grp.reset_index(drop=True)
        if len(grp) < total:
            continue
        arr_in  = grp[in_cols].values.astype(np.float32)
        arr_out = grp[out_cols].values.astype(np.float32)
        for i in range(0, len(grp) - total + 1, STRIDE):
            X_list.append(arr_in[i : i + seq_len])
            y_list.append(arr_out[i + seq_len : i + total])

    if not X_list:
        raise ValueError("No sequences - too little data per vessel")

    X = np.stack(X_list)
    y = np.stack(y_list)
    log.info(f"  Sequences: X={X.shape}, y={y.shape}")
    return X, y


# ═══════════════════════════════════════════════════════════════════
#  2. Dataset
# ═══════════════════════════════════════════════════════════════════

class AISDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════
#  3. LSTM model (same as lstm_predictor.py)
# ═══════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, n_features: int, n_targets: int,
                 hidden: int = HIDDEN, num_layers: int = NUM_LAYERS,
                 pred_len: int = PRED_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.pred_len  = pred_len
        self.n_targets = n_targets
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, pred_len * n_targets),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        pred   = self.fc(last)
        return pred.view(-1, self.pred_len, self.n_targets)


# ═══════════════════════════════════════════════════════════════════
#  4. Training (same as lstm_predictor.py)
# ═══════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, epochs, lr, patience, device, results_dir,
          model_filename="lstm_model_temporal.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve    = 0
    train_losses, val_losses = [], []
    model_path = Path(results_dir) / model_filename

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                v_loss += criterion(model(X_b), y_b).item()
        v_loss /= len(val_loader)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improve    = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Best val_loss: {best_val_loss:.4f}  →  {model_path}")
    return train_losses, val_losses, model_path


# ═══════════════════════════════════════════════════════════════════
#  5. Metrics (same as lstm_predictor.py)
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, loader, scaler_y, device, target_names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = model(X_b.to(device)).cpu().numpy()
            preds.append(p)
            trues.append(y_b.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    B, T, C = preds.shape
    preds_flat = preds.reshape(-1, C)
    trues_flat = trues.reshape(-1, C)

    if scaler_y is not None:
        preds_flat = scaler_y.inverse_transform(preds_flat)
        trues_flat = scaler_y.inverse_transform(trues_flat)

    metrics = {}
    for i, name in enumerate(target_names):
        mae  = np.mean(np.abs(preds_flat[:, i] - trues_flat[:, i]))
        rmse = np.sqrt(np.mean((preds_flat[:, i] - trues_flat[:, i]) ** 2))
        metrics[name] = {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4)}

    return metrics, preds_flat.reshape(B, T, C), trues_flat.reshape(B, T, C)


# ═══════════════════════════════════════════════════════════════════
#  6. Plots
# ═══════════════════════════════════════════════════════════════════

def plot_loss(train_losses, val_losses, figures_dir, filename="fig_loss_curve_temporal.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train loss", color="#2196F3")
    ax.plot(val_losses,   label="Val loss",   color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Training Curves (temporal split)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(figures_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Loss curve → {path}")


def plot_predictions(preds, trues, target_names, figures_dir,
                     n_samples=4, filename="fig_predictions_temporal.png"):
    fig, axes = plt.subplots(len(target_names), n_samples,
                             figsize=(4 * n_samples, 3 * len(target_names)))
    if len(target_names) == 1:
        axes = axes[np.newaxis, :]

    idx = np.random.choice(len(preds), n_samples, replace=False)

    for row, name in enumerate(target_names):
        for col, s in enumerate(idx):
            ax = axes[row, col]
            x  = np.arange(preds.shape[1])
            mae_val = np.mean(np.abs(preds[s, :, row] - trues[s, :, row]))
            ax.plot(x, trues[s, :, row], "o--", color="#2196F3", label="True")
            ax.plot(x, preds[s, :, row], "s-",  color="#F44336", label="Pred")
            ax.set_title(f"{name} | sample {col+1}\nMAE={mae_val:.4f}", fontsize=8)
            ax.set_xlabel("Steps ahead")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("LSTM predictions vs ground truth (temporal split test set)", fontsize=10)
    fig.tight_layout()
    path = Path(figures_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Predictions → {path}")


# ═══════════════════════════════════════════════════════════════════
#  7. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir",  default=AIS_MERGED_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--check",       action="store_true", help="Only check the data")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(FIGURES_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Loading the raw data ──
    df_raw = load_data(args.merged_dir)

    if args.check:
        log.info("=== Data check ===")
        log.info(f"Rows: {len(df_raw):,}")
        ts_check = pd.to_datetime(df_raw["BaseDateTime"], utc=True, errors="coerce")
        log.info(f"Period: {ts_check.min()} → {ts_check.max()}")
        log.info(f"Columns: {list(df_raw.columns)}")
        for col in TARGET_FEATURES:
            if col in df_raw.columns:
                log.info(f"  {col}: min={df_raw[col].min():.3f}  max={df_raw[col].max():.3f}"
                         f"  mean={df_raw[col].mean():.3f}  std={df_raw[col].std():.3f}")
        return

    # ── Temporal split BEFORE preprocessing and windowing ──
    # IMPORTANT: the split is made on the raw data so that preprocessing
    # cannot accidentally filter out whole months
    df_train_raw, df_val_raw, df_test_raw = temporal_split(df_raw)

    # ── Clean each part separately ──
    log.info("Cleaning train...")
    df_train = preprocess(df_train_raw)
    log.info("Cleaning val...")
    df_val   = preprocess(df_val_raw)
    log.info("Cleaning test...")
    df_test  = preprocess(df_test_raw)

    # ── Cut the sequences separately for each part ──
    log.info("Cutting train sequences...")
    X_train, y_train = make_sequences(df_train, SEQ_LEN, PRED_LEN)

    log.info("Cutting val sequences...")
    X_val, y_val     = make_sequences(df_val,   SEQ_LEN, PRED_LEN)

    log.info("Cutting test sequences...")
    X_test, y_test   = make_sequences(df_test,  SEQ_LEN, PRED_LEN)

    in_cols  = [c for c in INPUT_FEATURES  if c in df_train.columns]
    out_cols = [c for c in TARGET_FEATURES if c in df_train.columns]
    del df_raw, df_train_raw, df_val_raw, df_test_raw, df_train, df_val, df_test   # free memory
    gc.collect()

    # ── Normalisation: fit on train ONLY ──
    # Essential: the scaler does not see the val/test data during training
    B_tr, T_in, F_in   = X_train.shape
    B_tr, T_out, F_out  = y_train.shape

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_norm = scaler_x.fit_transform(
        X_train.reshape(-1, F_in)).reshape(X_train.shape).astype(np.float32)
    y_train_norm = scaler_y.fit_transform(
        y_train.reshape(-1, F_out)).reshape(y_train.shape).astype(np.float32)

    # transform (not fit_transform) for val and test
    X_val_norm  = scaler_x.transform(X_val.reshape(-1, F_in)).reshape(X_val.shape).astype(np.float32)
    y_val_norm  = scaler_y.transform(y_val.reshape(-1, F_out)).reshape(y_val.shape).astype(np.float32)

    X_test_norm = scaler_x.transform(X_test.reshape(-1, F_in)).reshape(X_test.shape).astype(np.float32)
    y_test_norm = scaler_y.transform(y_test.reshape(-1, F_out)).reshape(y_test.shape).astype(np.float32)

    del X_train, X_val   # the normalised copies are used from here on
    gc.collect()

    joblib.dump(scaler_x, results_dir / "scaler_x_temporal.pkl")
    joblib.dump(scaler_y, results_dir / "scaler_y_temporal.pkl")

    log.info(f"Sizes after normalisation:")
    log.info(f"  Train: X={X_train_norm.shape}  y={y_train_norm.shape}")
    log.info(f"  Val:   X={X_val_norm.shape}    y={y_val_norm.shape}")
    log.info(f"  Test:  X={X_test_norm.shape}   y={y_test_norm.shape}")

    # ── DataLoaders ──
    train_loader = DataLoader(AISDataset(X_train_norm, y_train_norm),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(AISDataset(X_val_norm,   y_val_norm),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(AISDataset(X_test_norm,  y_test_norm),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Model ──
    model = LSTMPredictor(
        n_features=len(in_cols),
        n_targets=len(out_cols),
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
        dropout=DROPOUT,
    ).to(device)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──
    train_losses, val_losses, model_path = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=LR, patience=PATIENCE,
        device=device, results_dir=results_dir,
        model_filename="lstm_model_temporal.pt",
    )

    # ── Load the best model ──
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ── Test evaluation ──
    metrics, preds, trues = evaluate(model, test_loader, scaler_y, device, out_cols)

    log.info("\n=== TEST METRICS (temporal split, June) ===")
    rows = []
    for name, m in metrics.items():
        log.info(f"  {name}: MAE={m['MAE']}  RMSE={m['RMSE']}")
        rows.append({"target": name, **m})

    metrics_df = pd.DataFrame(rows)
    metrics_path = results_dir / "metrics_lstm_temporal.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Metrics → {metrics_path}")

    # ── Trivial baselines on the same test sequences (original units) ──
    # persistence: the last observed value is kept for the whole horizon;
    # train mean: the mean of the training targets
    idx_in  = [in_cols.index(c) for c in out_cols]
    persist = np.repeat(X_test[:, -1:, idx_in], PRED_LEN, axis=1)
    clim    = np.broadcast_to(y_train.reshape(-1, F_out).mean(axis=0), y_test.shape)
    rows_b = []
    for bname, pred in [("LSTM", preds), ("persistence", persist), ("train mean", clim)]:
        for k, name in enumerate(out_cols):
            for step in ["all"] + list(range(1, PRED_LEN + 1)):
                e = (pred[:, :, k] - y_test[:, :, k]) if step == "all" else (pred[:, step - 1, k] - y_test[:, step - 1, k])
                rows_b.append({"model": bname, "target": name, "step": step,
                               "MAE": round(float(np.abs(e).mean()), 4),
                               "RMSE": round(float(np.sqrt((e ** 2).mean())), 4)})
    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(results_dir / "metrics_baselines_temporal.csv", index=False)
    log.info("\n=== TEST MAE: LSTM vs baselines (all steps) ===")
    log.info("\n" + df_b[df_b["step"] == "all"].pivot(index="model", columns="target", values="MAE").to_string())

    # ── Plots ──
    plot_loss(train_losses, val_losses, figures_dir)
    plot_predictions(preds, trues, out_cols, figures_dir)

    log.info("\n✅ Done! Files:")
    log.info(f"  Model:   {model_path}")
    log.info(f"  Metrics: {metrics_path}")
    log.info(f"  Figures: {figures_dir}")
    log.info("\nCompare metrics_lstm_temporal.csv with metrics_lstm.csv")
    log.info("to see the effect of the temporal split on MAE.")


if __name__ == "__main__":
    main()
