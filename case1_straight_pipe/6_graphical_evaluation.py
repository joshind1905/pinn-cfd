import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob

# ============================================================
# CONFIG
# ============================================================

DATA_FOLDER = "data_files"
VALIDATION_FOLDER = "3_sims_validation"
MODEL_FILE = os.path.join(DATA_FOLDER, "pinn_channel_phys_parametric.pt")

# Create output folder for plots in main directory (not in data_files)
PLOTS_FOLDER = "u(y)+ΔP plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_FOLDER}")

x_mid_user = 0.1
tol_user = 1.0e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. LOAD MODEL AND SCALING INFO
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
# 2. BUILD MODEL
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
# 3. FIND ALL VALIDATION CASES
# ============================================================

case_folders = glob.glob(os.path.join(VALIDATION_FOLDER, "case_*"))
print(f"\n=== Found {len(case_folders)} validation cases ===")

all_u_slices = []
all_y_slices = []
all_u_preds = []
all_x_values_list = []
all_delta_p_cfd_list = []
all_delta_p_pinn_list = []

for case_folder in sorted(case_folders):
    sim_label = os.path.basename(case_folder)
    print(f"\nProcessing {sim_label}")
    
    # Find CSV file in this case folder
    csv_files = glob.glob(os.path.join(case_folder, "*.csv"))
    if not csv_files:
        print(f"  Warning: No CSV file found in {case_folder}, skipping")
        continue
    
    CSV_FILE = csv_files[0]
    
    # ============================================================
    # LOAD CFD DATA
    # ============================================================
    
    df = pd.read_csv(CSV_FILE)
    df_sim = df.copy()
    
    x_np = df_sim["x"].values
    y_np = df_sim["y"].values
    u_np = df_sim["u"].values
    p_np = df_sim["p"].values
    
    # ============================================================
    # CHOOSE SECTION x-LOCATION (robust) for u(y)
    # ============================================================
    
    x_mid = x_mid_user
    tol = tol_user
    mask = np.abs(x_np - x_mid) < tol
    
    if mask.sum() == 0:
        x_min_sim = x_np.min()
        x_max_sim = x_np.max()
        x_mid = 0.5 * (x_min_sim + x_max_sim)
        tol = 0.05 * (x_max_sim - x_min_sim)
        print(f"  Using automatic section at x ≈ {x_mid:.3e}")
        mask = np.abs(x_np - x_mid) < tol
    
    if mask.sum() == 0:
        print(f"  Warning: No points found for u(y) section in {sim_label}, skipping")
        continue
    
    x_slice = x_np[mask]
    y_slice = y_np[mask]
    u_slice = u_np[mask]
    
    order = np.argsort(y_slice)
    x_slice = x_slice[order]
    y_slice = y_slice[order]
    u_slice = u_slice[order]
    
    # ============================================================
    # GET u(y) PREDICTIONS
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
    
    # ============================================================
    # PRESSURE DROP ANALYSIS ΔP(x)
    # ============================================================
    
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
            p_cfd_at_x = df_sim.loc[mask_x, "p"].values.mean()
            
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
    
    # Calculate pressure drop
    delta_p_cfd_x = x_pressure_cfd[0] - x_pressure_cfd
    delta_p_pinn_x = x_pressure_pinn[0] - x_pressure_pinn
    
    # Store results
    all_u_slices.append(u_slice)
    all_y_slices.append(y_slice)
    all_u_preds.append(u_pred)
    all_x_values_list.append(x_values)
    all_delta_p_cfd_list.append(delta_p_cfd_x)
    all_delta_p_pinn_list.append(delta_p_pinn_x)

# ============================================================
# 4. CALCULATE MEAN VALUES
# ============================================================

print(f"\n{'='*60}")
print(f"CALCULATING MEAN VALUES ACROSS {len(all_u_slices)} CASES")
print(f"{'='*60}")

# For u(y), interpolate to common y points
y_min_common = max([y.min() for y in all_y_slices])
y_max_common = min([y.max() for y in all_y_slices])
y_common = np.linspace(y_min_common, y_max_common, 100)

# Interpolate each case to common y points
u_slice_interp = []
u_pred_interp = []

for i in range(len(all_u_slices)):
    u_slice_interp.append(np.interp(y_common, all_y_slices[i], all_u_slices[i]))
    u_pred_interp.append(np.interp(y_common, all_y_slices[i], all_u_preds[i]))

# Calculate mean
u_slice_mean = np.mean(u_slice_interp, axis=0)
u_pred_mean = np.mean(u_pred_interp, axis=0)

# For ΔP(x), find common x points
x_min_common = max([x.min() for x in all_x_values_list])
x_max_common = min([x.max() for x in all_x_values_list])
x_common = np.linspace(x_min_common, x_max_common, 50)

# Interpolate each case to common x points
delta_p_cfd_interp = []
delta_p_pinn_interp = []

for i in range(len(all_x_values_list)):
    delta_p_cfd_interp.append(np.interp(x_common, all_x_values_list[i], all_delta_p_cfd_list[i]))
    delta_p_pinn_interp.append(np.interp(x_common, all_x_values_list[i], all_delta_p_pinn_list[i]))

# Calculate mean
delta_p_cfd_mean = np.mean(delta_p_cfd_interp, axis=0)
delta_p_pinn_mean = np.mean(delta_p_pinn_interp, axis=0)

# ============================================================
# 5. CREATE THE TWO PLOTS WITH MEAN VALUES
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: u(y) velocity profile - MEAN values
ax1.plot(u_slice_mean, y_common, "k-", label="CFD (mean)", linewidth=2)
ax1.plot(u_pred_mean, y_common, "r--", label="PINN (mean)", linewidth=2)

ax1.set_xlabel("u [m/s]", fontsize=12)
ax1.set_ylabel("y [m]", fontsize=12)
ax1.set_title(f"Mean Velocity Profile u(y) at x ≈ {x_mid_user:.3e}", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: ΔP(x) pressure drop - MEAN values
ax2.plot(x_common, delta_p_cfd_mean, "k-", label="CFD (mean)", linewidth=2)
ax2.plot(x_common, delta_p_pinn_mean, "r--", label="PINN (mean)", linewidth=2)

ax2.set_xlabel("x [m]", fontsize=12)
ax2.set_ylabel("ΔP [Pa] (from inlet)", fontsize=12)
ax2.set_title("Mean Pressure Drop ΔP(x)", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure in the plots folder
filename = f"u(y)+ΔP_plots_MEAN_across_{len(all_u_slices)}_cases.png"
filepath = os.path.join(PLOTS_FOLDER, filename)
plt.savefig(filepath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {filepath}")

plt.show()

print(f"\n✓ Analysis complete. Mean plots saved to '{PLOTS_FOLDER}/'")
