import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

DATA_FOLDER = "data_files"
CSV_FILE = os.path.join(DATA_FOLDER, "cfd_data_val_fluidfoam.csv")
MODEL_FILE = os.path.join(DATA_FOLDER, "pinn_channel_phys_parametric.pt")

# Create output folder for plots in main directory (not in data_files)
PLOTS_FOLDER = "u(y)+ΔP plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_FOLDER}")

SIM_ID = 30
x_mid_user = 0.1
tol_user = 1.0e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. LOAD CFD DATA (VALIDATION CSV) & SELECT ONE SIMULATION
# ============================================================

df = pd.read_csv(CSV_FILE)
print(f"\n=== CFD Data Overview ===")
print(f"Total CFD points in file: {len(df)}")
print(f"x range: [{df['x'].min():.3e}, {df['x'].max():.3e}]")
print(f"y range: [{df['y'].min():.3e}, {df['y'].max():.3e}]")
print(f"Available simulations: {np.unique(df['sim'].values)}")

sim_int = int(SIM_ID)
candidate_sim = f"sim_{sim_int}"
candidate_case = f"case_{sim_int:03d}"
available = df["sim"].unique()

if candidate_sim in available:
    sim_label = candidate_sim
elif candidate_case in available:
    sim_label = candidate_case
else:
    raise ValueError(
        f"Could not find a match for SIM_ID={sim_int}. "
        f"Tried '{candidate_sim}' and '{candidate_case}'. "
        f"Available labels: {available}"
    )

print(f"\nSelected simulation: {sim_label}")
df_sim = df[df["sim"] == sim_label].copy()
if df_sim.empty:
    raise ValueError(f"No points found for sim == {sim_label}")

print(f"Points in simulation: {len(df_sim)}")

x_np = df_sim["x"].values
y_np = df_sim["y"].values
u_np = df_sim["u"].values
p_np = df_sim["p"].values

# ============================================================
# 1a. CHOOSE SECTION x-LOCATION (robust) for u(y)
# ============================================================

x_mid = x_mid_user
tol = tol_user
mask = np.abs(x_np - x_mid) < tol

if mask.sum() == 0:
    x_min_sim = x_np.min()
    x_max_sim = x_np.max()
    x_mid = 0.5 * (x_min_sim + x_max_sim)
    tol = 0.05 * (x_max_sim - x_min_sim)
    print(f"\n[Info] No points found near x = {x_mid_user:.3e} with tol = {tol_user:.1e}")
    print(f"        Using automatic section at x ≈ {x_mid:.3e} (±{tol:.1e})")
    mask = np.abs(x_np - x_mid) < tol

if mask.sum() == 0:
    raise ValueError(
        f"No points found even with automatic section. "
        f"x range is [{x_np.min():.3e}, {x_np.max():.3e}]."
    )

x_slice = x_np[mask]
y_slice = y_np[mask]
u_slice = u_np[mask]

print(f"\nPoints in u(y) section: {len(x_slice)}")
order = np.argsort(y_slice)
x_slice = x_slice[order]
y_slice = y_slice[order]
u_slice = u_slice[order]

# ============================================================
# 2. LOAD MODEL AND SCALING INFO
# ============================================================

checkpoint = torch.load(MODEL_FILE, map_location=device)

# Get scaling info
x_min, x_max = checkpoint["x_min"], checkpoint["x_max"]
y_min, y_max = checkpoint["y_min"], checkpoint["y_max"]

if "u_scale" in checkpoint:
    u_scale = checkpoint["u_scale"]
    u_min = 0.0
    u_max = u_scale
    p_scale = checkpoint["p_scale"]
    p_min = 0.0
    p_max = p_scale
else:
    u_min, u_max = checkpoint["u_min"], checkpoint["u_max"]
    p_min, p_max = checkpoint["p_min"], checkpoint["p_max"]
    u_scale = u_max - u_min
    p_scale = p_max - p_min

param_cols = checkpoint["param_cols"]
param_mins = checkpoint["param_mins"]
param_maxs = checkpoint["param_maxs"]

def norm(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-12)

def denorm(arr_n, mn, mx):
    return arr_n * (mx - mn + 1e-12) + mn

# ============================================================
# 3. BUILD MODEL
# ============================================================

class PINN(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        layer_indices = sorted([int(k.split('.')[1]) for k, v in state_dict.items() 
                               if 'weight' in k and 'net' in k])
        
        layers = []
        for i, layer_idx in enumerate(layer_indices):
            weight_key = f"net.{layer_idx}.weight"
            bias_key = f"net.{layer_idx}.bias"
            
            weight = state_dict[weight_key]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight = nn.Parameter(weight.clone())
            linear.bias = nn.Parameter(state_dict[bias_key].clone())
            
            layers.append(linear)
            if i < len(layer_indices) - 1:
                layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, y, params):
        inp = torch.cat([x, y, params], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

model = PINN(checkpoint["model_state_dict"]).to(device)
model.eval()

# ============================================================
# 4. GET u(y) PREDICTIONS
# ============================================================

x_slice_n = norm(x_slice, x_min, x_max)
y_slice_n = norm(y_slice, y_min, y_max)

x_t = torch.tensor(x_slice_n, dtype=torch.float32, device=device).view(-1, 1)
y_t = torch.tensor(y_slice_n, dtype=torch.float32, device=device).view(-1, 1)

param_tensors = []
for col in param_cols:
    arr = df_sim.loc[mask, col].values
    arr_n = norm(arr, param_mins[col], param_maxs[col])
    param_tensors.append(
        torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
    )
params_t = torch.cat(param_tensors, dim=1)

with torch.no_grad():
    u_pred_n, _, _ = model(x_t, y_t, params_t)

u_pred = denorm(u_pred_n.cpu().numpy().ravel(), u_min, u_max)

# Calculate u(y) errors
err_u = u_pred - u_slice
rmse_u = np.sqrt(np.mean(err_u ** 2))
abs_error_u = np.mean(np.abs(err_u))
rel_error_u = rmse_u / (np.max(np.abs(u_slice)) + 1e-12) * 100

# ============================================================
# 5. PRESSURE DROP ANALYSIS ΔP(x) - MODIFIED RMSE CALCULATION
# ============================================================

print(f"\n{'='*60}")
print("PRESSURE DROP ANALYSIS ΔP(x)")
print(f"{'='*60}")

# Get the full x-range for this simulation (same as second script)
x_min_sim = x_np.min()
x_max_sim = x_np.max()

# Create multiple x-slices for pressure drop calculation
n_slices = 20
x_unique = np.sort(df_sim["x"].unique())
x_positions = np.linspace(x_unique.min(), x_unique.max(), n_slices)

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
        
        x_pts_n = norm(x_pts, x_min, x_max)
        y_pts_n = norm(y_pts, y_min, y_max)
        
        x_t = torch.tensor(x_pts_n, dtype=torch.float32, device=device).view(-1, 1)
        y_t = torch.tensor(y_pts_n, dtype=torch.float32, device=device).view(-1, 1)
        
        param_tensors_pts = []
        for col in param_cols:
            arr = df_sim.loc[mask_x, col].values
            arr_n = norm(arr, param_mins[col], param_maxs[col])
            param_tensors_pts.append(
                torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
            )
        params_t_pts = torch.cat(param_tensors_pts, dim=1)
        
        with torch.no_grad():
            _, _, p_pred_n_pts = model(x_t, y_t, params_t_pts)
        
        p_pred_pts = denorm(p_pred_n_pts.cpu().numpy().ravel(), p_min, p_max)
        p_pinn_at_x = p_pred_pts.mean()
        
        x_pressure_pinn.append(p_pinn_at_x)
        x_pressure_cfd.append(p_cfd_at_x)
        x_values.append(x_pos)

x_values = np.array(x_values)
x_pressure_cfd = np.array(x_pressure_cfd)
x_pressure_pinn = np.array(x_pressure_pinn)

# Calculate pressure drop from inlet (ΔP(x) = P_inlet - P(x))
p_inlet_cfd = x_pressure_cfd[0]
p_inlet_pinn = x_pressure_pinn[0]

delta_p_cfd_x = p_inlet_cfd - x_pressure_cfd
delta_p_pinn_x = p_inlet_pinn - x_pressure_pinn

# Calculate total pressure drop (inlet to outlet)
delta_p_cfd_total = delta_p_cfd_x[-1]
delta_p_pinn_total = delta_p_pinn_x[-1]
delta_p_total_error = delta_p_pinn_total - delta_p_cfd_total
delta_p_total_error_percent = abs(delta_p_total_error)/abs(delta_p_cfd_total)*100 if abs(delta_p_cfd_total) > 0 else 0

# ============================================================
# 5b. CALCULATE RMSE FOR ΔP POINT-BY-POINT
# ============================================================

# Calculate error for ΔP at each x position (exactly like u error at each y)
delta_p_error = delta_p_pinn_x - delta_p_cfd_x

# Same RMSE calculation as u(y)
rmse_delta_p = np.sqrt(np.mean(delta_p_error ** 2))
abs_error_delta_p = np.mean(np.abs(delta_p_error))

# Same relative error calculation as u(y) - using max of CFD values
rel_error_delta_p = rmse_delta_p / (np.max(np.abs(delta_p_cfd_x)) + 1e-12) * 100

# ============================================================
# 6. PRINT RESULTS SUMMARY - UPDATED WITH NEW RMSE
# ============================================================

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\nSimulation: {sim_label}")

print(f"\n--- u(y) at x ≈ {x_mid:.3e} ---")
print(f"  RMSE:        {rmse_u:.3e} m/s")
print(f"  Mean Abs Error: {abs_error_u:.3e} m/s")
print(f"  Relative RMSE:  {rel_error_u:.2f}%")

print(f"\n--- ΔP(x) Pressure Drop ---")
print(f"  Total ΔP (inlet to outlet):")
print(f"    CFD:  {delta_p_cfd_total:.4f} Pa")
print(f"    PINN: {delta_p_pinn_total:.4f} Pa")
print(f"    Error: {delta_p_total_error:+.4f} Pa ({delta_p_total_error_percent:.2f}%)")
print(f"\n  ΔP(x) along pipe (entire curve):")
print(f"    RMSE: {rmse_delta_p:.3e} Pa ({rel_error_delta_p:.2f}%)")
print(f"    Mean Abs Error:  {abs_error_delta_p:.3e} Pa")

# ============================================================
# 7. CREATE THE TWO PLOTS - UPDATED WITH NEW ΔP RMSE
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: u(y) velocity profile - NO MARKERS, just lines
ax1.plot(u_slice, y_slice, "k-", label="CFD", linewidth=2)
ax1.plot(u_pred, y_slice, "r--", label="PINN", linewidth=2)
ax1.set_xlabel("u [m/s]", fontsize=12)
ax1.set_ylabel("y [m]", fontsize=12)
ax1.set_title(f"Velocity Profile u(y) at x ≈ {x_mid:.3e}", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Add u error annotation
ax1.text(0.05, 0.95, f"RMSE = {rmse_u:.3e} m/s\n({rel_error_u:.2f}%)", 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: ΔP(x) pressure drop - just lines
ax2.plot(x_values, delta_p_cfd_x, "k-", label="CFD", linewidth=2)
ax2.plot(x_values, delta_p_pinn_x, "r--", label="PINN", linewidth=2)
ax2.set_xlabel("x [m]", fontsize=12)
ax2.set_ylabel("ΔP [Pa] (from inlet)", fontsize=12)
ax2.set_title(f"Pressure Drop ΔP(x) - {sim_label}", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Add ΔP error annotation - UPDATED to show curve RMSE
ax2.text(0.05, 0.95, f"Curve RMSE = {rmse_delta_p:.3e} Pa\n({rel_error_delta_p:.2f}%)", 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure in the plots folder
filename = f"u(y)+ΔP_plots_{sim_label}_x{x_mid:.3f}.png"
filepath = os.path.join(PLOTS_FOLDER, filename)
plt.savefig(filepath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {filepath}")

plt.show()

print(f"\n✓ Analysis complete. All plots saved to '{PLOTS_FOLDER}/'")
