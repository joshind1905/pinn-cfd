import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Data folder where CSV files are stored
DATA_FOLDER = "data_files"

# Use the correct CSV files for training and validation (now in data_files)
TRAIN_CSV_FILE = os.path.join(DATA_FOLDER, "cfd_data_train_fluidfoam.csv")
VAL_CSV_FILE   = os.path.join(DATA_FOLDER, "cfd_data_val_fluidfoam.csv")

N_EPOCHS = 4000 #4000-----------------------------------------------------------------
LR = 1e-4
PRINT_EVERY = 500
BATCH_SIZE = None  # None = full batch

# Loss weights
LAMBDA_PDE = 1e-3
LAMBDA_DATA = 1.0

# Parameter columns used as additional inputs - UPDATED to match your DOE
PARAM_COLS = ["U_ave", "kin_vis", "H", "L", "h", "l"]
KIN_VIS_COL = "kin_vis"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. LOAD CFD DATA (TRAIN + VAL) + PARAMETERS
# ============================================================

# Check if files exist
if not os.path.exists(TRAIN_CSV_FILE):
    raise FileNotFoundError(f"Training file not found: {TRAIN_CSV_FILE}")
if not os.path.exists(VAL_CSV_FILE):
    raise FileNotFoundError(f"Validation file not found: {VAL_CSV_FILE}")

df_train = pd.read_csv(TRAIN_CSV_FILE)
df_val   = pd.read_csv(VAL_CSV_FILE)

print("TRAIN CSV:", TRAIN_CSV_FILE, "| points:", len(df_train))
print("VAL   CSV:", VAL_CSV_FILE,   "| points:", len(df_val))

# ============================================================
# FILTER OUT BAD SIMULATIONS
# ============================================================
print("\n=== FILTERING OUT BAD SIMULATIONS ===")

def filter_simulations(df, threshold=1e6):
    """Remove simulations with values above threshold"""
    sims_to_keep = []
    sims_to_remove = []
    
    for sim in df['sim'].unique():
        df_sim = df[df['sim'] == sim]
        
        # Check if this simulation has reasonable values
        u_max = np.abs(df_sim['u']).max()
        v_max = np.abs(df_sim['v']).max()
        p_max = np.abs(df_sim['p']).max()
        
        if u_max < threshold and v_max < threshold and p_max < threshold:
            sims_to_keep.append(sim)
        else:
            sims_to_remove.append(sim)
    
    print(f"Keeping {len(sims_to_keep)} simulations")
    print(f"Removing {len(sims_to_remove)} bad simulations with values > {threshold}")
    if sims_to_remove:
        print(f"Removed simulations: {sims_to_remove[:10]}...")
    
    return df[df['sim'].isin(sims_to_keep)]

# Filter training data
df_train_filtered = filter_simulations(df_train)
df_val_filtered = filter_simulations(df_val)

print(f"\nTraining points after filtering: {len(df_train_filtered)}")
print(f"Validation points after filtering: {len(df_val_filtered)}")

# Use filtered data
df_train = df_train_filtered
df_val = df_val_filtered

# ============================================================
# NORMALIZE PER SIMULATION (NOT GLOBAL)
# ============================================================
print("\n=== NORMALIZING PER SIMULATION ===")

def normalize_simulation_data(df):
    """Normalize each simulation individually"""
    x_norm_list = []
    y_norm_list = []
    u_norm_list = []
    v_norm_list = []
    p_norm_list = []
    
    # Store scaling factors per simulation
    sim_scales = {}
    
    for sim in df['sim'].unique():
        mask = df['sim'] == sim
        df_sim = df[mask]
        
        # Get min/max for this simulation
        x_min, x_max = df_sim['x'].min(), df_sim['x'].max()
        y_min, y_max = df_sim['y'].min(), df_sim['y'].max()
        u_min, u_max = df_sim['u'].min(), df_sim['u'].max()
        v_min, v_max = df_sim['v'].min(), df_sim['v'].max()
        p_min, p_max = df_sim['p'].min(), df_sim['p'].max()
        
        # Store scales
        sim_scales[sim] = {
            'x': (x_min, x_max), 'y': (y_min, y_max),
            'u': (u_min, u_max), 'v': (v_min, v_max), 'p': (p_min, p_max)
        }
        
        # Normalize to [0, 1]
        x_norm = (df_sim['x'].values - x_min) / (x_max - x_min + 1e-12)
        y_norm = (df_sim['y'].values - y_min) / (y_max - y_min + 1e-12)
        u_norm = (df_sim['u'].values - u_min) / (u_max - u_min + 1e-12)
        v_norm = (df_sim['v'].values - v_min) / (v_max - v_min + 1e-12)
        p_norm = (df_sim['p'].values - p_min) / (p_max - p_min + 1e-12)
        
        x_norm_list.append(x_norm)
        y_norm_list.append(y_norm)
        u_norm_list.append(u_norm)
        v_norm_list.append(v_norm)
        p_norm_list.append(p_norm)
    
    # Create normalized dataframe
    df_norm = df.copy()
    df_norm['x'] = np.concatenate(x_norm_list)
    df_norm['y'] = np.concatenate(y_norm_list)
    df_norm['u'] = np.concatenate(u_norm_list)
    df_norm['v'] = np.concatenate(v_norm_list)
    df_norm['p'] = np.concatenate(p_norm_list)
    
    return df_norm, sim_scales

# Normalize training and validation data
print("Normalizing training data...")
df_train_norm, train_scales = normalize_simulation_data(df_train)
print("Normalizing validation data...")
df_val_norm, val_scales = normalize_simulation_data(df_val)

# Use normalized data for training
x_train_n = df_train_norm["x"].values.astype(np.float32)
y_train_n = df_train_norm["y"].values.astype(np.float32)
u_train_n = df_train_norm["u"].values.astype(np.float32)
v_train_n = df_train_norm["v"].values.astype(np.float32)
p_train_n = df_train_norm["p"].values.astype(np.float32)

x_val_n = df_val_norm["x"].values.astype(np.float32)
y_val_n = df_val_norm["y"].values.astype(np.float32)
u_val_n = df_val_norm["u"].values.astype(np.float32)
v_val_n = df_val_norm["v"].values.astype(np.float32)
p_val_n = df_val_norm["p"].values.astype(np.float32)

# Parameters (these are physical values, not normalized)
params_train_phys = {col: df_train[col].values.astype(np.float32) for col in PARAM_COLS}
params_val_phys   = {col: df_val[col].values.astype(np.float32)   for col in PARAM_COLS}

# Kinematic viscosity (physical)
nu_train_phys = df_train[KIN_VIS_COL].values.astype(np.float32)
nu_val_phys   = df_val[KIN_VIS_COL].values.astype(np.float32)

print(f"\nNormalization complete!")
print(f"Training points after filtering: {len(df_train)}")
print(f"Validation points after filtering: {len(df_val)}")

# Check normalized ranges
print(f"\nNormalized data ranges:")
print(f"x_train_n: [{x_train_n.min():.3f}, {x_train_n.max():.3f}]")
print(f"y_train_n: [{y_train_n.min():.3f}, {y_train_n.max():.3f}]")
print(f"u_train_n: [{u_train_n.min():.3f}, {u_train_n.max():.3f}]")
print(f"v_train_n: [{v_train_n.min():.3f}, {v_train_n.max():.3f}]")
print(f"p_train_n: [{p_train_n.min():.3f}, {p_train_n.max():.3f}]")

# ============================================================
# 2. NORMALIZE PARAMETERS (GLOBAL - THESE ARE CLEAN)
# ============================================================

def normalize(arr, mn=None, mx=None):
    """Normalize array to [0, 1] range"""
    if mn is None:
        mn = arr.min()
    if mx is None:
        mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-12), mn, mx

def apply_normalization(arr, mn, mx):
    """Apply existing normalization to array"""
    return (arr - mn) / (mx - mn + 1e-12)

# Parameters normalized (using TRAIN stats only)
param_mins = {}
param_maxs = {}
param_train_n_list = []
param_val_n_list = []

for col in PARAM_COLS:
    # TRAIN
    arr_train = params_train_phys[col]
    arr_train_n, mn, mx = normalize(arr_train)
    param_mins[col] = mn
    param_maxs[col] = mx
    param_train_n_list.append(
        torch.tensor(arr_train_n, dtype=torch.float32, device=device).view(-1, 1)
    )

    # VAL (using train mn, mx)
    arr_val = params_val_phys[col]
    arr_val_n = apply_normalization(arr_val, mn, mx)
    param_val_n_list.append(
        torch.tensor(arr_val_n, dtype=torch.float32, device=device).view(-1, 1)
    )

params_train_n_tensor = torch.cat(param_train_n_list, dim=1)
params_val_n_tensor   = torch.cat(param_val_n_list,   dim=1)

# ============================================================
# 3. BUILD TENSORS (NOW WITH NORMALIZED CFD DATA)
# ============================================================

# TRAIN tensors
x_train = torch.tensor(x_train_n, dtype=torch.float32, device=device).view(-1, 1)
y_train = torch.tensor(y_train_n, dtype=torch.float32, device=device).view(-1, 1)
u_train_n_t = torch.tensor(u_train_n, dtype=torch.float32, device=device).view(-1, 1)
v_train_n_t = torch.tensor(v_train_n, dtype=torch.float32, device=device).view(-1, 1)
p_train_n_t = torch.tensor(p_train_n, dtype=torch.float32, device=device).view(-1, 1)
nu_train_t  = torch.tensor(nu_train_phys, dtype=torch.float32, device=device).view(-1, 1)

# VAL tensors
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
print(f"Number of parameters: {N_PARAMS}")
print(f"Parameter list: {PARAM_COLS}")

# ============================================================
# 4. PARAMETRIC PINN ARCHITECTURE
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

    def forward(self, x, y, params):
        inp = torch.cat([x, y, params], dim=1)
        out = self.net(inp)
        u_n = out[:, 0:1]
        v_n = out[:, 1:2]
        p_n = out[:, 2:3]
        return u_n, v_n, p_n

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 5. AUTOGRAD HELPER
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

# ============================================================
# 6. TRAINING LOOP (MODIFIED - NO DENORMALIZATION NEEDED)
# ============================================================

print("\nStarting training...")

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # Full batch
    x_in_n = x_train.clone().detach().requires_grad_(True)
    y_in_n = y_train.clone().detach().requires_grad_(True)
    params_in_n = params_train_n_tensor

    u_data_batch = u_train_n_t
    v_data_batch = v_train_n_t
    p_data_batch = p_train_n_t
    nu_batch = nu_train_t

    # Forward pass (predictions are already normalized)
    u_pred_n, v_pred_n, p_pred_n = model(x_in_n, y_in_n, params_in_n)

    # Data loss
    loss_data = torch.mean(
        (u_pred_n - u_data_batch) ** 2 +
        (v_pred_n - v_data_batch) ** 2 +
        (p_pred_n - p_data_batch) ** 2
    )

    # Compute derivatives (still in normalized space)
    du_dx = grad(u_pred_n, x_in_n)
    du_dy = grad(u_pred_n, y_in_n)
    dv_dx = grad(v_pred_n, x_in_n)
    dv_dy = grad(v_pred_n, y_in_n)
    dp_dx = grad(p_pred_n, x_in_n)
    dp_dy = grad(p_pred_n, y_in_n)

    # Second derivatives
    d2u_dx2 = grad(du_dx, x_in_n)
    d2u_dy2 = grad(du_dy, y_in_n)
    d2v_dx2 = grad(dv_dx, x_in_n)
    d2v_dy2 = grad(dv_dy, y_in_n)

    # PDE loss in normalized space (simpler)
    r_cont = du_dx + dv_dy
    
    # Note: This is a simplified PDE loss. For physical accuracy,
    # you'd need to properly scale the equations.
    loss_pde = torch.mean(r_cont ** 2 + du_dx**2 + dv_dy**2)

    # Total loss
    total_loss = LAMBDA_PDE * loss_pde + LAMBDA_DATA * loss_data

    total_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        u_val_pred_n, v_val_pred_n, p_val_pred_n = model(x_val, y_val, params_val_n_tensor)
        val_loss_data = torch.mean(
            (u_val_pred_n - u_val_n_t) ** 2 +
            (v_val_pred_n - v_val_n_t) ** 2 +
            (p_val_pred_n - p_val_n_t) ** 2
        )

    # Logging
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d} | "
            f"Train Loss: {total_loss.item():.3e} | "
            f"PDE: {loss_pde.item():.3e} | "
            f"Data: {loss_data.item():.3e} | "
            f"Val Data: {val_loss_data.item():.3e}"
        )

print("\nTraining completed!")

# ============================================================
# 7. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "param_cols": PARAM_COLS,
    "param_mins": param_mins,
    "param_maxs": param_maxs,
    # Note: We don't save spatial/output scales since we normalized per simulation
}

torch.save(checkpoint, "models/pinn_channel_phys_parametric.pt")
print("\nModel saved to 'models/pinn_channel_phys_parametric.pt'")
print(f"Parameter columns: {PARAM_COLS}")
