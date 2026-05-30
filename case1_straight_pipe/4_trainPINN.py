import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Set data folder for CFD data
DATA_FOLDER = "data_files"
DOE_FOLDER = "2_DOE"  # Folder containing DOE files

TRAIN_CSV_FILE = os.path.join(DATA_FOLDER, "cfd_data_train_fluidfoam.csv")
VAL_CSV_FILE   = os.path.join(DATA_FOLDER, "cfd_data_val_fluidfoam.csv")

N_EPOCHS = 4000
LR = 1e-3
PRINT_EVERY = 500
BATCH_SIZE = None

# Loss weights - CRITICAL FIX: Force v to zero
LAMBDA_PDE = 1e-3
LAMBDA_DATA = 1.0
LAMBDA_V_ZERO = 100.0  # Strong penalty for non-zero v

# Parameter columns
PARAM_COLS = ["U_ave", "kin_vis"]
KIN_VIS_COL = "kin_vis"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Reading CFD data from folder: {DATA_FOLDER}")
print(f"DOE files located in: {DOE_FOLDER}")

# Check if files exist
if not os.path.exists(TRAIN_CSV_FILE):
    raise FileNotFoundError(f"Training file not found: {TRAIN_CSV_FILE}")
if not os.path.exists(VAL_CSV_FILE):
    raise FileNotFoundError(f"Validation file not found: {VAL_CSV_FILE}")

# ============================================================
# 1. LOAD DATA
# ============================================================

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_val   = pd.read_csv(VAL_CSV_FILE)

print("TRAIN CSV:", TRAIN_CSV_FILE, "| points:", len(df_train))
print("VAL   CSV:", VAL_CSV_FILE,   "| points:", len(df_val))

# Extract data
x_train_phys = df_train["x"].values.astype(np.float32)
y_train_phys = df_train["y"].values.astype(np.float32)
u_train_phys = df_train["u"].values.astype(np.float32)
v_train_phys = df_train["v"].values.astype(np.float32)
p_train_phys = df_train["p"].values.astype(np.float32)

x_val_phys = df_val["x"].values.astype(np.float32)
y_val_phys = df_val["y"].values.astype(np.float32)
u_val_phys = df_val["u"].values.astype(np.float32)
v_val_phys = df_val["v"].values.astype(np.float32)
p_val_phys = df_val["p"].values.astype(np.float32)

# Parameters
params_train_phys = {col: df_train[col].values.astype(np.float32) for col in PARAM_COLS}
params_val_phys   = {col: df_val[col].values.astype(np.float32)   for col in PARAM_COLS}
nu_train_phys = df_train[KIN_VIS_COL].values.astype(np.float32)
nu_val_phys   = df_val[KIN_VIS_COL].values.astype(np.float32)

# ============================================================
# 2. NORMALIZATION
# ============================================================

def normalize(arr, mn=None, mx=None):
    if mn is None:
        mn = arr.min()
    if mx is None:
        mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-12), mn, mx

def apply_normalization(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-12)

# Spatial normalisation
x_train_n, x_min, x_max = normalize(x_train_phys)
y_train_n, y_min, y_max = normalize(y_train_phys)
x_val_n = apply_normalization(x_val_phys, x_min, x_max)
y_val_n = apply_normalization(y_val_phys, y_min, y_max)

# Output normalisation - CRITICAL: v should be centered at 0 as the flow is not vertical
u_train_n, u_min, u_max = normalize(u_train_phys)
# For v: We want it to predict exactly 0, so normalize differently
v_train_n, v_min, v_max = normalize(v_train_phys)
# Force v_min and v_max to be symmetric around 0 to encourage zero predictions
v_max = max(abs(v_min), abs(v_max))
v_min = -v_max
v_train_n = v_train_phys / (v_max + 1e-12)  # This maps [-v_max, v_max] to [-1, 1]

p_train_n, p_min, p_max = normalize(p_train_phys)

# Apply same scaling to VAL
u_val_n = apply_normalization(u_val_phys, u_min, u_max)
v_val_n = v_val_phys / (v_max + 1e-12)  # Same symmetric scaling
p_val_n = apply_normalization(p_val_phys, p_min, p_max)

# Store scales
x_scale = x_max - x_min
y_scale = y_max - y_min
u_scale = u_max - u_min
v_scale = v_max  # This is now the half-range
p_scale = p_max - p_min

print(f"\nScaling factors:")
print(f"x_scale: {x_scale:.3e}, y_scale: {y_scale:.3e}")
print(f"u_scale: {u_scale:.3e}, v_scale: {v_scale:.3e}, p_scale: {p_scale:.3e}")
print(f"v range: [{v_min:.3e}, {v_max:.3e}]")

# Parameter normalisation
param_mins = {}
param_maxs = {}
param_train_n_list = []
param_val_n_list = []

for col in PARAM_COLS:
    arr_train = params_train_phys[col]
    arr_train_n, mn, mx = normalize(arr_train)
    param_mins[col] = mn
    param_maxs[col] = mx
    param_train_n_list.append(
        torch.tensor(arr_train_n, dtype=torch.float32, device=device).view(-1, 1)
    )
    arr_val = params_val_phys[col]
    arr_val_n = apply_normalization(arr_val, mn, mx)
    param_val_n_list.append(
        torch.tensor(arr_val_n, dtype=torch.float32, device=device).view(-1, 1)
    )

params_train_n_tensor = torch.cat(param_train_n_list, dim=1)
params_val_n_tensor   = torch.cat(param_val_n_list,   dim=1)

# ============================================================
# 3. BUILD TENSORS
# ============================================================

# Training tensors
x_train = torch.tensor(x_train_n, dtype=torch.float32, device=device).view(-1, 1)
y_train = torch.tensor(y_train_n, dtype=torch.float32, device=device).view(-1, 1)
u_train_n_t = torch.tensor(u_train_n, dtype=torch.float32, device=device).view(-1, 1)
v_train_n_t = torch.tensor(v_train_n, dtype=torch.float32, device=device).view(-1, 1)
p_train_n_t = torch.tensor(p_train_n, dtype=torch.float32, device=device).view(-1, 1)
nu_train_t  = torch.tensor(nu_train_phys, dtype=torch.float32, device=device).view(-1, 1)

# Validation tensors
x_val = torch.tensor(x_val_n, dtype=torch.float32, device=device).view(-1, 1)
y_val = torch.tensor(y_val_n, dtype=torch.float32, device=device).view(-1, 1)
u_val_n_t = torch.tensor(u_val_n, dtype=torch.float32, device=device).view(-1, 1)
v_val_n_t = torch.tensor(v_val_n, dtype=torch.float32, device=device).view(-1, 1)
p_val_n_t = torch.tensor(p_val_n, dtype=torch.float32, device=device).view(-1, 1)
nu_val_t  = torch.tensor(nu_val_phys, dtype=torch.float32, device=device).view(-1, 1)

N_train = x_train.shape[0]
N_val   = x_val.shape[0]
N_PARAMS = params_train_n_tensor.shape[1]

print(f"\nNumber of training points: {N_train}")
print(f"Number of validation points: {N_val}")

# ============================================================
# 4. PINN ARCHITECTURE
# ============================================================

class PINN(nn.Module):
    def __init__(self, in_dim=2 + N_PARAMS, out_dim=3, width=64, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize the last layer to output small values
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.1)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, x, y, params):
        inp = torch.cat([x, y, params], dim=1)
        out = self.net(inp)
        
        # CRITICAL FIX: Add a small L2 regularization to force v to zero
        # But still allow it to be learned if needed
        u_n = out[:, 0:1]
        v_n = out[:, 1:2] * 0.1  # Scale down v predictions by 10x initially
        p_n = out[:, 2:3]
        return u_n, v_n, p_n

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 5. AUTOGRAD HELPER
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

# ============================================================
# 6. TRAINING LOOP - WITH STRONG V ZERO CONSTRAINT
# ============================================================

x_scale_t = torch.tensor(x_scale, dtype=torch.float32, device=device)
y_scale_t = torch.tensor(y_scale, dtype=torch.float32, device=device)

print("\nStarting training...")
print(f"LAMBDA_V_ZERO = {LAMBDA_V_ZERO}")

# Store predictions for plotting
predictions_history = []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # Batch selection
    if BATCH_SIZE is not None and BATCH_SIZE < N_train:
        idx = torch.randint(0, N_train, (BATCH_SIZE,), device=device)
        x_in_n = x_train[idx].clone().detach().requires_grad_(True)
        y_in_n = y_train[idx].clone().detach().requires_grad_(True)
        params_in_n = params_train_n_tensor[idx]
        u_data_batch = u_train_n_t[idx]
        v_data_batch = v_train_n_t[idx]
        p_data_batch = p_train_n_t[idx]
        nu_batch = nu_train_t[idx]
    else:
        x_in_n = x_train.clone().detach().requires_grad_(True)
        y_in_n = y_train.clone().detach().requires_grad_(True)
        params_in_n = params_train_n_tensor
        u_data_batch = u_train_n_t
        v_data_batch = v_train_n_t
        p_data_batch = p_train_n_t
        nu_batch = nu_train_t

    # Forward pass
    u_pred_n, v_pred_n, p_pred_n = model(x_in_n, y_in_n, params_in_n)

    # ========== DATA LOSSES ==========
    loss_u = torch.mean((u_pred_n - u_data_batch) ** 2)
    loss_v = torch.mean((v_pred_n - v_data_batch) ** 2)
    loss_p = torch.mean((p_pred_n - p_data_batch) ** 2)
    
    # ========== ADDITIONAL V ZERO CONSTRAINT ==========
    loss_v_zero = torch.mean(v_pred_n ** 2)
    
    # Combined data loss
    loss_data = loss_u + loss_v + loss_p + LAMBDA_V_ZERO * loss_v_zero

    # PDE loss
    u_pred_phys = u_pred_n * u_scale + u_min
    v_pred_phys = v_pred_n * v_scale  
    p_pred_phys = p_pred_n * p_scale + p_min

    # Derivatives
    du_dx_n = grad(u_pred_phys, x_in_n)
    du_dy_n = grad(u_pred_phys, y_in_n)
    dv_dx_n = grad(v_pred_phys, x_in_n)
    dv_dy_n = grad(v_pred_phys, y_in_n)
    dp_dx_n = grad(p_pred_phys, x_in_n)
    dp_dy_n = grad(p_pred_phys, y_in_n)

    # Scale derivatives
    du_dx = du_dx_n / x_scale_t
    du_dy = du_dy_n / y_scale_t
    dv_dx = dv_dx_n / x_scale_t
    dv_dy = dv_dy_n / y_scale_t
    dp_dx = dp_dx_n / x_scale_t
    dp_dy = dp_dy_n / y_scale_t

    # Second derivatives
    d2u_dx2_n = grad(du_dx_n, x_in_n)
    d2u_dy2_n = grad(du_dy_n, y_in_n)
    d2v_dx2_n = grad(dv_dx_n, x_in_n)
    d2v_dy2_n = grad(dv_dy_n, y_in_n)

    d2u_dx2 = d2u_dx2_n / (x_scale_t ** 2)
    d2u_dy2 = d2u_dy2_n / (y_scale_t ** 2)
    d2v_dx2 = d2v_dx2_n / (x_scale_t ** 2)
    d2v_dy2 = d2v_dy2_n / (y_scale_t ** 2)

    # Continuity equation
    r_cont = du_dx + dv_dy

    # Momentum equations
    r_mom_u = (
        u_pred_phys * du_dx +
        v_pred_phys * du_dy +
        dp_dx -
        nu_batch * (d2u_dx2 + d2u_dy2)
    )

    r_mom_v = (
        u_pred_phys * dv_dx +
        v_pred_phys * dv_dy +
        dp_dy -
        nu_batch * (d2v_dx2 + d2v_dy2)
    )

    loss_pde = torch.mean(r_cont ** 2 + r_mom_u ** 2 + r_mom_v ** 2)

    # Total loss
    total_loss = LAMBDA_PDE * loss_pde + loss_data

    total_loss.backward()
    
    # Gradient clipping to prevent oscillations
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    # Validation and save predictions at certain epochs
    if epoch % 500 == 0 or epoch == N_EPOCHS:
        model.eval()
        with torch.no_grad():
            # Get predictions for a fixed x-slice (e.g., mid-channel)
            x_mid_val = 0.5 * (x_val_phys.min() + x_val_phys.max())
            mask_slice = np.abs(x_val_phys - x_mid_val) < 0.05 * (x_val_phys.max() - x_val_phys.min())
            
            if mask_slice.sum() > 0:
                y_slice = y_val_phys[mask_slice]
                x_slice = x_val_phys[mask_slice]
                
                # Normalize
                x_slice_n = apply_normalization(x_slice, x_min, x_max)
                y_slice_n = apply_normalization(y_slice, y_min, y_max)
                
                x_t = torch.tensor(x_slice_n, dtype=torch.float32, device=device).view(-1, 1)
                y_t = torch.tensor(y_slice_n, dtype=torch.float32, device=device).view(-1, 1)
                
                # Get parameters for this slice
                param_slice_tensors = []
                for col in PARAM_COLS:
                    arr = df_val.loc[mask_slice, col].values
                    arr_n = apply_normalization(arr, param_mins[col], param_maxs[col])
                    param_slice_tensors.append(
                        torch.tensor(arr_n, dtype=torch.float32, device=device).view(-1, 1)
                    )
                params_slice_t = torch.cat(param_slice_tensors, dim=1)
                
                # Predict
                u_pred_n, v_pred_n, p_pred_n = model(x_t, y_t, params_slice_t)
                
                # Denormalise
                u_pred = (u_pred_n * u_scale + u_min).cpu().numpy().ravel()
                p_pred = (p_pred_n * p_scale + p_min).cpu().numpy().ravel()
                
                # Get CFD values
                u_cfd = u_val_phys[mask_slice][np.argsort(y_slice)]
                p_cfd = p_val_phys[mask_slice][np.argsort(y_slice)]
                y_sorted = np.sort(y_slice)
                
                # Store
                predictions_history.append({
                    'epoch': epoch,
                    'y': y_sorted,
                    'u_cfd': u_cfd,
                    'u_pinn': u_pred[np.argsort(y_slice)],
                    'p_cfd': p_cfd,
                    'p_pinn': p_pred[np.argsort(y_slice)]
                })

    # Logging
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss: {total_loss.item():.3e} | "
            f"PDE: {loss_pde.item():.3e} | "
            f"U: {loss_u.item():.3e} | "
            f"V: {loss_v.item():.3e} | "
            f"V-zero: {loss_v_zero.item():.3e} | "
            f"P: {loss_p.item():.3e}"
        )

print("\nTraining completed!")

# ============================================================
# 7. SAVE MODEL
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    # Spatial scaling
    "x_min": x_min, "x_max": x_max,
    "y_min": y_min, "y_max": y_max,
    # Output scaling
    "u_min": u_min, "u_max": u_max,
    "v_min": -v_max, "v_max": v_max, 
    "p_min": p_min, "p_max": p_max,
    # Parametric scaling
    "param_cols": PARAM_COLS,
    "param_mins": param_mins,
    "param_maxs": param_maxs,

}

# Save in current directory
model_path = os.path.join(DATA_FOLDER, "pinn_channel_phys_parametric.pt")
torch.save(checkpoint, model_path)
print(f"\nModel saved to '{model_path}'")
print("\nTraining script completed successfully!")
