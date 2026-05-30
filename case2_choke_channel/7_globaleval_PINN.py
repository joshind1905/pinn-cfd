#!/usr/bin/env python3
"""
5_eval_PINN.py

Evaluate the quality of the PINN predictions on a chosen dataset
(train or validation), using the CFD data as ground truth.

- Loads:
    * models/pinn_channel_phys_parametric.pt   (trained model)
    * data_files/cfd_data_val_fluidfoam.csv   (by default) or train CSV

- Processes ALL simulations in the dataset
- Computes per-simulation metrics:
    * RMSE(u) - velocity field error
    * relative RMSE(u) - normalized by max absolute velocity at that slice
    * RMSE(ΔP) - pressure drop along pipe error
    * relative RMSE(ΔP) - normalized by total pressure drop
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

# Data and model folders
DATA_FOLDER = "data_files"
MODELS_FOLDER = "models"

# Choose split: "val" or "train"
SPLIT = "val"         # "val" for validation, "train" for training

# Dataset CSVs
TRAIN_CSV_FILE = os.path.join(DATA_FOLDER, "cfd_data_train_fluidfoam.csv")
VAL_CSV_FILE   = os.path.join(DATA_FOLDER, "cfd_data_val_fluidfoam.csv")

# Model checkpoint
MODEL_FILE = os.path.join(MODELS_FOLDER, "pinn_channel_phys_parametric.pt")

# Network architecture (based on checkpoint)
WIDTH = 64
DEPTH = 7   # Checkpoint shows 7 linear layers (net.0 to net.12)

# Number of slices for pressure drop analysis
N_PRESSURE_SLICES = 20

# Output metrics file
OUTPUT_METRICS_FILE = f"pinn_eval_{SPLIT}_metrics.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. CHOOSE DATASET & LOAD CSV
# ============================================================

if SPLIT == "train":
    CSV_FILE = TRAIN_CSV_FILE
elif SPLIT == "val":
    CSV_FILE = VAL_CSV_FILE
else:
    raise ValueError("SPLIT must be 'train' or 'val'.")

# Check if files exist
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

print(f"Evaluating PINN on {SPLIT.upper()} dataset: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
print("Total points:", len(df))
all_simulations = df["sim"].unique()
print(f"Total simulations in dataset: {len(all_simulations)}")

# Basic checks
required_cols = ["x", "y", "u", "p"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {CSV_FILE}.")

# ============================================================
# 2. LOAD CHECKPOINT & EXTRACT PARAM INFO
# ============================================================

checkpoint = torch.load(MODEL_FILE, map_location=device)

# Get parametric info from checkpoint
param_cols = checkpoint["param_cols"]
param_mins = checkpoint["param_mins"]
param_maxs = checkpoint["param_maxs"]
N_PARAMS = len(param_cols)

print("\nParametric columns:", param_cols)
for col in param_cols:
    print(f"  {col}: [{param_mins[col]:.3e}, {param_maxs[col]:.3e}]")

# Check parameters exist in CSV
for col in param_cols:
    if col not in df.columns:
        raise KeyError(f"Parameter column '{col}' not found in {CSV_FILE}.")

# ============================================================
# 3. NORMALIZATION FUNCTIONS
# ============================================================

def norm(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-12)

def denorm(arr_n, mn, mx):
    return arr_n * (mx - mn + 1e-12) + mn

# ============================================================
# 4. REBUILD PINN
# ============================================================

class PINN(nn.Module):
    """
    Same architecture as in training script
    """
    def __init__(self, in_dim=2 + N_PARAMS, out_dim=3, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 2):
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
        return u_n, v_n, p_n  # Return all 3 values

# Create model
model = PINN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("\nModel loaded successfully!")

# ============================================================
# 5. PROCESS EACH SIMULATION INDIVIDUALLY
# ============================================================

sims = df["sim"].unique()
rows = []
failed_sims = []

print(f"\nProcessing all {len(sims)} simulations...")

for sim_idx, sim in enumerate(sims):
    if sim_idx % 10 == 0:
        print(f"  Progress: {sim_idx}/{len(sims)} simulations processed")
    
    try:
        # Get data for this simulation
        mask = df["sim"] == sim
        df_sim = df[mask].copy()
        
        if len(df_sim) == 0:
            failed_sims.append(sim)
            continue
        
        # Get physical coordinates and values for this simulation
        x_phys = df_sim["x"].values
        y_phys = df_sim["y"].values
        u_phys = df_sim["u"].values
        p_phys = df_sim["p"].values
        
        # Use this simulation's min/max for normalization
        x_min, x_max = x_phys.min(), x_phys.max()
        y_min, y_max = y_phys.min(), y_phys.max()
        u_min, u_max = u_phys.min(), u_phys.max()
        p_min, p_max = p_phys.min(), p_phys.max()
        
        # Normalize spatial coordinates using this simulation's min/max
        x_n = norm(x_phys, x_min, x_max)
        y_n = norm(y_phys, y_min, y_max)
        
        x_t = torch.tensor(x_n, dtype=torch.float32, device=device).view(-1, 1)
        y_t = torch.tensor(y_n, dtype=torch.float32, device=device).view(-1, 1)
        
        # Normalize parameters using stored mins/maxs from checkpoint (global)
        param_tensors = []
        for col in param_cols:
            arr = df_sim[col].values
            mn = param_mins[col]
            mx = param_maxs[col]
            arr_n = norm(arr, mn, mx)
            param_tensors.append(
                torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
            )
        
        params_t = torch.cat(param_tensors, dim=1)  # [N, N_PARAMS]
        
        # Run PINN inference - get all 3 outputs
        with torch.no_grad():
            u_pred_n, v_pred_n, p_pred_n = model(x_t, y_t, params_t)
        
        # Denormalize using this simulation's ranges
        u_pred = denorm(u_pred_n.cpu().numpy().ravel(), u_min, u_max)
        p_pred = denorm(p_pred_n.cpu().numpy().ravel(), p_min, p_max)
        
        # ============================================================
        # 5a. COMPUTE u METRICS - EXACTLY like 6_pinnpostprocessing.py
        # ============================================================
        
        err_u = u_pred - u_phys
        rmse_u = np.sqrt(np.mean(err_u**2))
        max_abs_u = np.max(np.abs(u_phys)) + 1e-12
        rel_rmse_u = rmse_u / max_abs_u * 100  # Multiply by 100 for percentage
        
        # ============================================================
        # 5b. COMPUTE PRESSURE DROP ΔP METRICS - EXACTLY like 6_pinnpostprocessing.py
        # ============================================================
        
        # Create multiple x-slices along pipe length for pressure drop analysis
        x_unique = np.sort(df_sim["x"].unique())
        x_positions = np.linspace(x_unique.min(), x_unique.max(), N_PRESSURE_SLICES)
        
        x_pressure_pinn = []
        x_pressure_cfd = []
        x_values = []
        
        for x_pos in x_positions:
            tol_x = 0.05 * (x_unique.max() - x_unique.min())
            mask_x = np.abs(df_sim["x"].values - x_pos) < tol_x
            
            if mask_x.sum() > 0:
                # CFD pressure
                p_cfd_at_x = df_sim.loc[mask_x, "p"].values.mean()
                
                # PINN predictions
                x_pts = df_sim.loc[mask_x, "x"].values
                y_pts = df_sim.loc[mask_x, "y"].values
                
                # Normalize using this simulation's range
                x_pts_n = (x_pts - x_min) / (x_max - x_min + 1e-12)
                y_pts_n = (y_pts - y_min) / (y_max - y_min + 1e-12)
                
                x_t = torch.tensor(x_pts_n, dtype=torch.float32, device=device).view(-1, 1)
                y_t = torch.tensor(y_pts_n, dtype=torch.float32, device=device).view(-1, 1)
                
                param_tensors_pts = []
                for col in param_cols:
                    arr = df_sim.loc[mask_x, col].values
                    arr_n = (arr - param_mins[col]) / (param_maxs[col] - param_mins[col] + 1e-12)
                    param_tensors_pts.append(
                        torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
                    )
                params_t_pts = torch.cat(param_tensors_pts, dim=1)
                
                with torch.no_grad():
                    u_pred_n_pts, v_pred_n_pts, p_pred_n_pts = model(x_t, y_t, params_t_pts)
                
                # Denormalize pressure using this simulation's range
                p_pred_pts = p_pred_n_pts.cpu().numpy().ravel() * (p_max - p_min + 1e-12) + p_min
                p_pinn_at_x = p_pred_pts.mean()
                
                x_pressure_pinn.append(p_pinn_at_x)
                x_pressure_cfd.append(p_cfd_at_x)
                x_values.append(x_pos)
        
        if len(x_values) > 2:  # Need at least 3 points for meaningful pressure drop
            x_values = np.array(x_values)
            x_pressure_cfd = np.array(x_pressure_cfd)
            x_pressure_pinn = np.array(x_pressure_pinn)
            
            # Calculate pressure drop from inlet - EXACTLY like 6_pinnpostprocessing.py
            delta_p_cfd_x = x_pressure_cfd[0] - x_pressure_cfd
            delta_p_pinn_x = x_pressure_pinn[0] - x_pressure_pinn
            
            # Total pressure drop (inlet to outlet)
            delta_p_cfd_total = x_pressure_cfd[0] - x_pressure_cfd[-1]
            delta_p_pinn_total = x_pressure_pinn[0] - x_pressure_pinn[-1]
            
            # Calculate ΔP errors - EXACTLY like 6_pinnpostprocessing.py
            delta_p_error = delta_p_pinn_total - delta_p_cfd_total
            delta_p_error_percent = abs(delta_p_error) / abs(delta_p_cfd_total) * 100
            rmse_delta_p = np.sqrt(np.mean((delta_p_pinn_x - delta_p_cfd_x) ** 2))
            abs_error_delta_p = np.mean(np.abs(delta_p_pinn_x - delta_p_cfd_x))
            
        else:
            # Not enough slices for pressure drop analysis
            rmse_delta_p = np.nan
            delta_p_error = np.nan
            delta_p_error_percent = np.nan
            delta_p_cfd_total = np.nan
            delta_p_pinn_total = np.nan
            abs_error_delta_p = np.nan
        
        # Store metrics for this simulation
        metrics = {
            "sim": sim,
            "N": len(df_sim),
            "RMSE_u": rmse_u,
            "rel_RMSE_u_percent": rel_rmse_u,  # Now as percentage
            "RMSE_deltaP": rmse_delta_p,
            "deltaP_error_percent": delta_p_error_percent,
            "deltaP_abs_error": delta_p_error,
            "deltaP_CFD_total": delta_p_cfd_total,
            "deltaP_PINN_total": delta_p_pinn_total,
            "deltaP_MAE": abs_error_delta_p
        }
        
        rows.append(metrics)
        
    except Exception as e:
        print(f"  Error processing simulation {sim}: {str(e)}")
        failed_sims.append(sim)

# ============================================================
# 6. CREATE METRICS DATAFRAME AND SAVE
# ============================================================

print(f"\nProcessing complete!")
print(f"Successfully processed: {len(rows)}/{len(sims)} simulations")
if failed_sims:
    print(f"Failed simulations: {failed_sims}")

df_metrics = pd.DataFrame(rows)

# Save metrics file
out_csv = Path(OUTPUT_METRICS_FILE).resolve()
df_metrics.to_csv(out_csv, index=False)

# Also save a backup to data_files
backup_csv = os.path.join(DATA_FOLDER, OUTPUT_METRICS_FILE)
df_metrics.to_csv(backup_csv, index=False)

print(f"\n✅ All metrics saved to: {out_csv}")
print(f"✅ Backup metrics saved to: {backup_csv}")
print(f"\nTotal simulations in output file: {len(df_metrics)}")

# Show ALL simulations in the output
print("\n=== ALL PER-SIMULATION METRICS ===")
pd.set_option('display.float_format', '{:.3e}'.format)
pd.set_option('display.max_rows', None)  # Show all rows
if len(df_metrics) > 0:
    print(df_metrics[['sim', 'N', 'RMSE_u', 'rel_RMSE_u_percent', 'RMSE_deltaP', 'deltaP_error_percent', 'deltaP_abs_error']])
else:
    print("No simulations were successfully processed.")

# Show summary of valid pressure drop calculations
if len(df_metrics) > 0:
    valid_dp = df_metrics['RMSE_deltaP'].notna().sum()
    print(f"\nPressure drop calculated for {valid_dp}/{len(df_metrics)} simulations")
else:
    print(f"\nNo simulations were successfully processed.")

print(f"\n[SUCCESS] All {len(df_metrics)} simulations evaluated!")
