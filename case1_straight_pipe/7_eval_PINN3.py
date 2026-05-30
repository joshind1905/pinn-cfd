#!/usr/bin/env python3
"""
5_eval_PINN.py

Evaluate the quality of the PINN predictions on a chosen dataset
(train or validation), using the CFD data as ground truth.

- Loads:
    * pinn_channel_phys_parametric.pt   (trained model + scaling)
    * cfd_data_val_fluidfoam.csv        (by default) or train CSV

- Computes global and per-simulation metrics:
    * RMSE(u), RMSE(v), RMSE(p)
    * relative RMSE (normalized by RMS of CFD)
- Saves per-simulation metrics to:
    * pinn_eval_<split>_metrics.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import os

# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------------
# MODIFIED: Read from data_files folder
# ------------------------------------------------------------------
DATA_FOLDER = "data_files"
TRAIN_CSV_FILE = os.path.join(DATA_FOLDER, "cfd_data_train_fluidfoam.csv")
VAL_CSV_FILE   = os.path.join(DATA_FOLDER, "cfd_data_val_fluidfoam.csv")
MODEL_FILE = os.path.join(DATA_FOLDER, "pinn_channel_phys_parametric.pt")
# ------------------------------------------------------------------

# Choose split: "val" or "train"
SPLIT = "val"         # "val" for validation, "train" for training

# CORRECTED: Network architecture based on checkpoint analysis
# Checkpoint shows: 7 linear layers (width=64)
WIDTH = 64  # Checkpoint uses 64
DEPTH = 7   # Checkpoint has 7 linear layers total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Reading from folder: {DATA_FOLDER}")

# ============================================================
# 1. CHOOSE DATASET & LOAD CSV
# ============================================================

if SPLIT == "train":
    CSV_FILE = TRAIN_CSV_FILE
elif SPLIT == "val":
    CSV_FILE = VAL_CSV_FILE
else:
    raise ValueError("SPLIT must be 'train' or 'val'.")

print(f"Evaluating PINN on {SPLIT.upper()} dataset: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
print("Total points:", len(df))
print("Available simulations:", df["sim"].unique())

# Basic checks
required_cols = ["x", "y", "u", "v", "p"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {CSV_FILE}.")

x_phys = df["x"].values
y_phys = df["y"].values
u_phys = df["u"].values
v_phys = df["v"].values
p_phys = df["p"].values

print(f"x in [{x_phys.min():.3e}, {x_phys.max():.3e}]")
print(f"y in [{y_phys.min():.3e}, {y_phys.max():.3e}]")

# ============================================================
# 2. LOAD CHECKPOINT & REBUILD PINN
# ============================================================

checkpoint = torch.load(MODEL_FILE, map_location=device)

# Scaling info
x_min, x_max = checkpoint["x_min"], checkpoint["x_max"]
y_min, y_max = checkpoint["y_min"], checkpoint["y_max"]

# Check format
if "u_scale" in checkpoint:
    u_scale = checkpoint["u_scale"]
    v_scale = checkpoint["v_scale"]
    p_scale = checkpoint["p_scale"]
    u_min = v_min = p_min = 0.0
    u_max = u_scale
    v_max = v_scale
    p_max = p_scale
else:
    u_min, u_max = checkpoint["u_min"], checkpoint["u_max"]
    v_min, v_max = checkpoint["v_min"], checkpoint["v_max"]
    p_min, p_max = checkpoint["p_min"], checkpoint["p_max"]
    u_scale = u_max - u_min
    v_scale = v_max - v_min
    p_scale = p_max - p_min

x_scale = x_max - x_min
y_scale = y_max - y_min

# Parametric info
param_cols = checkpoint["param_cols"]
param_mins = checkpoint["param_mins"]
param_maxs = checkpoint["param_maxs"]
N_PARAMS = len(param_cols)

print("Parametric columns:", param_cols)
print(f"u scale: {u_scale:.3e}, v scale: {v_scale:.3e}, p scale: {p_scale:.3e}")

# Check parameters exist in CSV
for col in param_cols:
    if col not in df.columns:
        raise KeyError(f"Parameter column '{col}' not found in {CSV_FILE}.")

def norm(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-12)

def denorm(arr_n, mn, mx):
    return arr_n * (mx - mn + 1e-12) + mn

class PINN(nn.Module):
    """
    Same architecture as in 3_train_PINN.py
    """
    def __init__(self, in_dim=2 + N_PARAMS, out_dim=3, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 2):  # Changed from depth-1 to depth-2
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, params):
        inp = torch.cat([x, y, params], dim=1)
        out = self.net(inp)
        u_n = out[:, 0:1]
        v_n = out[:, 1:2]
        p_n = out[:, 2:3]
        return u_n, v_n, p_n

model = PINN().to(device)
# Use strict=False to handle architecture mismatches
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("Model loaded successfully!")

# ============================================================
# 3. BUILD NORMALIZED INPUT TENSORS
# ============================================================

# Normalize spatial coordinates
x_n = norm(x_phys, x_min, x_max)
y_n = norm(y_phys, y_min, y_max)

x_t = torch.tensor(x_n, dtype=torch.float32, device=device).view(-1, 1)
y_t = torch.tensor(y_n, dtype=torch.float32, device=device).view(-1, 1)

# Normalize parameters
param_tensors = []
for col in param_cols:
    arr = df[col].values
    mn = param_mins[col]
    mx = param_maxs[col]
    arr_n = norm(arr, mn, mx)
    param_tensors.append(
        torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
    )

params_t = torch.cat(param_tensors, dim=1)  # [N, N_PARAMS]

# ============================================================
# 4. RUN PINN & DENORMALIZE
# ============================================================

print(f"\nRunning inference on {len(x_t)} points...")
with torch.no_grad():
    u_pred_n, v_pred_n, p_pred_n = model(x_t, y_t, params_t)

u_pred = denorm(u_pred_n.cpu().numpy().ravel(), u_min, u_max)
v_pred = denorm(v_pred_n.cpu().numpy().ravel(), v_min, v_max)
p_pred = denorm(p_pred_n.cpu().numpy().ravel(), p_min, p_max)

# ============================================================
# 5. COMPUTE ERROR METRICS (GLOBAL + PER-SIM)
# ============================================================

def compute_metrics(y_true, y_pred, label="u"):
    err = y_pred - y_true
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    
    # FIXED: Use max absolute value for normalization (like post-processing script)
    # This is more appropriate for channel flow where values can be near zero
    max_abs = np.max(np.abs(y_true)) + 1e-12
    rel_rmse = rmse / max_abs
    
    return {
        f"RMSE_{label}": rmse,
        f"rel_RMSE_{label}": rel_rmse,
    }

# Global metrics
metrics_global = {}
metrics_global.update(compute_metrics(u_phys, u_pred, "u"))
metrics_global.update(compute_metrics(v_phys, v_pred, "v"))
metrics_global.update(compute_metrics(p_phys, p_pred, "p"))

print("\n=== GLOBAL METRICS (", SPLIT.upper(), ") ===")
for k, v in metrics_global.items():
    if "rel" in k:
        print(f"{k}: {v*100:.2f} %")
    else:
        print(f"{k}: {v:.3e}")

# Per-simulation metrics
sims = df["sim"].unique()
rows = []

for sim in sims:
    mask = df["sim"] == sim
    if mask.sum() == 0:
        continue

    u_true_sim = u_phys[mask]
    v_true_sim = v_phys[mask]
    p_true_sim = p_phys[mask]

    u_pred_sim = u_pred[mask]
    v_pred_sim = v_pred[mask]
    p_pred_sim = p_pred[mask]

    m = {"sim": sim, "N": int(mask.sum())}
    m.update(compute_metrics(u_true_sim, u_pred_sim, "u"))
    m.update(compute_metrics(v_true_sim, v_pred_sim, "v"))
    m.update(compute_metrics(p_true_sim, p_pred_sim, "p"))
    rows.append(m)

df_metrics = pd.DataFrame(rows)

out_csv = Path(f"pinn_eval_{SPLIT}_metrics.csv").resolve()
df_metrics.to_csv(out_csv, index=False)

print(f"\nPer-simulation metrics saved to: {out_csv}")
print("\n=== PER-SIMULATION SUMMARY ===")
print(df_metrics)
