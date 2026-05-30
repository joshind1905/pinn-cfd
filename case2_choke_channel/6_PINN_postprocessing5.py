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
MODEL_FILE = os.path.join("models", "pinn_channel_phys_parametric.pt")

# Create output folder for plots in main directory (not in data_files)
PLOTS_FOLDER = "u(y)+ΔP plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_FOLDER}")

SIM_ID = 3
x_mid_user = 0.1  # This will be overridden by quarter points
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
# 1a. GET THE POINTS ALONG PIPE LENGTH
# ============================================================

# Get the full x-range for this simulation
x_min_sim = x_np.min()
x_max_sim = x_np.max()
x_range = x_max_sim - x_min_sim

# Define points (20%, 50%, 80% of pipe length) - CHANGED FROM QUARTERS (thats why the variables say quarters)
quarter_fractions = [0.2, 0.5, 0.8]
quarter_positions = [x_min_sim + f * x_range for f in quarter_fractions]

print(f"\n=== Quarter Points along Pipe Length ===")
print(f"Pipe length: {x_range:.3e} m")
print(f"Quarter positions:")
for i, pos in enumerate(quarter_positions):
    print(f"  {quarter_fractions[i]*100:.0f}%: x = {pos:.3e} m")

# ============================================================
# 2. LOAD MODEL AND SCALING INFO
# ============================================================

checkpoint = torch.load(MODEL_FILE, map_location=device)

# Print available keys for debugging
print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

# Get parameter scaling info (these are saved in checkpoint)
param_cols = checkpoint["param_cols"]
param_mins = checkpoint["param_mins"]
param_maxs = checkpoint["param_maxs"]

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
# 4. NORMALIZATION RANGES FOR THIS SIMULATION
# ============================================================

# For the selected simulation, get it's min/max for normalization
x_min_sim = x_np.min()
x_max_sim = x_np.max()
y_min_sim = y_np.min()
y_max_sim = y_np.max()
u_min_sim = u_np.min()
u_max_sim = u_np.max()
p_min_sim = p_np.min()
p_max_sim = p_np.max()

print(f"\nSimulation {sim_label} ranges:")
print(f"  x: [{x_min_sim:.3e}, {x_max_sim:.3e}]")
print(f"  y: [{y_min_sim:.3e}, {y_max_sim:.3e}]")
print(f"  u: [{u_min_sim:.3e}, {u_max_sim:.3e}]")
print(f"  p: [{p_min_sim:.3e}, {p_max_sim:.3e}]")

# ============================================================
# 5. CREATE PLOTS FOR EACH QUARTER POINT
# ============================================================

print(f"\n{'='*60}")
print(f"CREATING u(y) PLOTS AT QUARTER POINTS")
print(f"{'='*60}")

# Dictionary to store results for each quarter point
quarter_results = {}

for i, (fraction, x_pos) in enumerate(zip(quarter_fractions, quarter_positions)):
    print(f"\n--- Processing {fraction*100:.0f}% point (x = {x_pos:.3e}) ---")
    
    # Find points near this x position
    tol = 0.05 * x_range  # 5% of pipe length as tolerance
    mask = np.abs(x_np - x_pos) < tol
    
    if mask.sum() == 0:
        print(f"  WARNING: No points found near x = {x_pos:.3e}. Skipping.")
        continue
    
    x_slice = x_np[mask]
    y_slice = y_np[mask]
    u_slice = u_np[mask]
    
    print(f"  Points in section: {len(x_slice)}")
    order = np.argsort(y_slice)
    x_slice = x_slice[order]
    y_slice = y_slice[order]
    u_slice = u_slice[order]
    
    # Normalise the slice points using this simulation's range
    x_slice_n = (x_slice - x_min_sim) / (x_max_sim - x_min_sim + 1e-12)
    y_slice_n = (y_slice - y_min_sim) / (y_max_sim - y_min_sim + 1e-12)
    
    x_t = torch.tensor(x_slice_n, dtype=torch.float32, device=device).view(-1, 1)
    y_t = torch.tensor(y_slice_n, dtype=torch.float32, device=device).view(-1, 1)
    
    # Parameters - normalise using global mins/maxs from checkpoint
    param_tensors = []
    for col in param_cols:
        arr = df_sim.loc[mask, col].values
        arr_n = (arr - param_mins[col]) / (param_maxs[col] - param_mins[col] + 1e-12)
        param_tensors.append(
            torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
        )
    params_t = torch.cat(param_tensors, dim=1)
    
    # Get predictions
    with torch.no_grad():
        u_pred_n, _, _ = model(x_t, y_t, params_t)
    
    # Denormalise using this simulation's range
    u_pred = u_pred_n.cpu().numpy().ravel() * (u_max_sim - u_min_sim + 1e-12) + u_min_sim
    
    # Calculate u(y) errors
    err_u = u_pred - u_slice
    rmse_u = np.sqrt(np.mean(err_u ** 2))
    abs_error_u = np.mean(np.abs(err_u))
    rel_error_u = rmse_u / (np.max(np.abs(u_slice)) + 1e-12) * 100
    
    # Store results
    quarter_results[fraction] = {
        'x_pos': x_pos,
        'y_slice': y_slice,
        'u_slice': u_slice,
        'u_pred': u_pred,
        'rmse': rmse_u,
        'abs_error': abs_error_u,
        'rel_error': rel_error_u
    }
    
    print(f"  RMSE: {rmse_u:.3e} m/s ({rel_error_u:.2f}%)")

# ============================================================
# 6. PRESSURE DROP ANALYSIS ΔP(x) - KEEP ORIGINAL CURVE, FIX RMSE
# ============================================================

print(f"\n{'='*60}")
print("PRESSURE DROP ANALYSIS ΔP(x)")
print(f"{'='*60}")

# Create multiple x-slices for pressure drop calculation (KEEP ORIGINAL METHOD)
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
        
        # Normalize using this simulation's range
        x_pts_n = (x_pts - x_min_sim) / (x_max_sim - x_min_sim + 1e-12)
        y_pts_n = (y_pts - y_min_sim) / (y_max_sim - y_min_sim + 1e-12)
        
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
            _, _, p_pred_n_pts = model(x_t, y_t, params_t_pts)
        
        # Denormalize pressure using this simulation's range
        p_pred_pts = p_pred_n_pts.cpu().numpy().ravel() * (p_max_sim - p_min_sim + 1e-12) + p_min_sim
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

# ============================================================
# 6b. CALCULATE ERRORS EXACTLY LIKE u(y) - POINT-BY-POINT
# ============================================================

# Calculate error for ΔP at each x position (exactly like u error at each y)
delta_p_error = delta_p_pinn_x - delta_p_cfd_x

# Same RMSE calculation as u(y)
rmse_delta_p = np.sqrt(np.mean(delta_p_error ** 2))
abs_error_delta_p = np.mean(np.abs(delta_p_error))

# Same relative error calculation as u(y) - using max of CFD values
rel_error_delta_p = rmse_delta_p / (np.max(np.abs(delta_p_cfd_x)) + 1e-12) * 100

# Calculate total pressure drop error (inlet to outlet) - keep this too
delta_p_cfd_total = delta_p_cfd_x[-1]
delta_p_pinn_total = delta_p_pinn_x[-1]
delta_p_total_error = delta_p_pinn_total - delta_p_cfd_total
delta_p_total_error_percent = abs(delta_p_total_error)/abs(delta_p_cfd_total)*100
# ============================================================
# 7. PRINT RESULTS SUMMARY - UPDATED
# ============================================================

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\nSimulation: {sim_label}")

print(f"\n--- u(y) at Quarter Points ---")
for fraction, results in quarter_results.items():
    print(f"  {fraction*100:.0f}% (x = {results['x_pos']:.3e}):")
    print(f"    RMSE: {results['rmse']:.3e} m/s ({results['rel_error']:.2f}%)")

print(f"\n--- ΔP(x) Pressure Drop ---")
print(f"  Total ΔP (inlet to outlet):")
print(f"    CFD:  {delta_p_cfd_total:.4f} Pa")
print(f"    PINN: {delta_p_pinn_total:.4f} Pa")
print(f"    Error: {delta_p_total_error:+.4f} Pa ({delta_p_total_error_percent:.2f}%)")
print(f"\n  ΔP(x) along pipe (entire curve):")
print(f"    RMSE: {rmse_delta_p:.3e} Pa ({rel_error_delta_p:.2f}%)")
print(f"    Mean Abs Error: {abs_error_delta_p:.3e} Pa")

# ============================================================
# 8. CREATE THE PLOTS - UPDATED
# ============================================================

# Create a figure with subplots for each quarter point + pressure drop
n_quarter_plots = len(quarter_results)
fig, axes = plt.subplots(1, n_quarter_plots + 1, figsize=(5*(n_quarter_plots+1), 6))

# If only one quarter point found, axes might not be array
if n_quarter_plots == 1:
    axes = [axes[0], axes[1]]

# Plot u(y) for each quarter point
for idx, (fraction, results) in enumerate(sorted(quarter_results.items())):
    ax = axes[idx]
    
    ax.plot(results['u_slice'], results['y_slice'], "k-", label="CFD", linewidth=2)
    ax.plot(results['u_pred'], results['y_slice'], "r--", label="PINN", linewidth=2)
    ax.set_xlabel("u [m/s]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(f"u(y) at {fraction*100:.0f}%\nx={results['x_pos']:.3e}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add error annotation
    ax.text(0.05, 0.95, f"RMSE = {results['rmse']:.3e}\n({results['rel_error']:.2f}%)", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot ΔP(x) in the last subplot - UPDATED with correct error annotation
ax_dp = axes[-1]
ax_dp.plot(x_values, delta_p_cfd_x, "k-", label="CFD", linewidth=2)
ax_dp.plot(x_values, delta_p_pinn_x, "r--", label="PINN", linewidth=2)
ax_dp.set_xlabel("x [m]", fontsize=12)
ax_dp.set_ylabel("ΔP [Pa] (from inlet)", fontsize=12)
ax_dp.set_title(f"Pressure Drop ΔP(x)\n{sim_label}", fontsize=14)
ax_dp.legend(fontsize=12)
ax_dp.grid(True, alpha=0.3)

# Add ΔP error annotation - UPDATED to show curve RMSE
ax_dp.text(0.05, 0.95, f"Curve RMSE = {rmse_delta_p:.3e} Pa\n({rel_error_delta_p:.2f}%)", 
           transform=ax_dp.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()

# Save the figure in the plots folder
filename = f"quarter_points_ΔP_{sim_label}.png"
filepath = os.path.join(PLOTS_FOLDER, filename)
plt.savefig(filepath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {filepath}")

plt.show()

print(f"\n✓ Analysis complete. All plots saved to '{PLOTS_FOLDER}/'")
