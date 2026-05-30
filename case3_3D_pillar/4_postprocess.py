#!/usr/bin/env python3
"""
4_postprocess.py - Simple working version
"""

from pathlib import Path
import numpy as np
import pandas as pd
import fluidfoam as ff
import os


# Roots
TRAIN_ROOT = Path("3_sims_training").resolve()
VAL_ROOT = Path("3_sims_validation").resolve()

PARAM_COLS = ['channel_length', 'channel_width', 'channel_height', 
              'pillar_radius', 'U_ave', 'nu']

DATA_FOLDER = "data_files"
os.makedirs(DATA_FOLDER, exist_ok=True)


def parse_doe_row(doe_path: Path) -> dict:
    params = {}
    if not doe_path.is_file():
        return params
    for line in doe_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        line = line.replace(";", " ")
        tokens = line.split()
        if len(tokens) >= 2:
            try:
                params[tokens[0]] = float(tokens[1])
            except:
                params[tokens[0]] = tokens[1]
    return params


def add_dimensionless(params):
    out = dict(params)
    try:
        U = float(params.get("U_inlet", 0))
        R = float(params.get("pillar_radius", 0))
        nu = float(params.get("nu", 0))
        W = float(params.get("channel_width", 1))
        if U > 0 and R > 0 and nu > 0:
            D = 2 * R
            out["Re"] = U * D / nu
            out["pillar_diameter"] = D
            out["blockage_ratio"] = D / W
    except:
        pass
    return out


def get_n_cells(case_path):
    """Get number of cells from owner file"""
    owner_file = case_path / "constant" / "polyMesh" / "owner"
    if not owner_file.exists():
        return None
    
    with open(owner_file, 'r') as f:
        lines = f.readlines()
    
    # The owner file has header then one entry per face
    # Number of cells = max(owner value) + 1
    max_owner = -1
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if line and not line.startswith('('):
            try:
                # Owner file format: each line has an integer
                val = int(line.split()[0])
                if val > max_owner:
                    max_owner = val
            except:
                pass
    
    if max_owner >= 0:
        return max_owner + 1
    return None


def process_single_sim(sim_dir):
    case_name = sim_dir.name
    print(f"\n[INFO] {case_name}")
    
    # Get number of cells
    n_cells = get_n_cells(sim_dir)
    if n_cells is None:
        print(f"  Could not determine number of cells")
        return None
    print(f"  Number of cells: {n_cells}")
    
    # Get cell centers using fluidfoam
    try:
        centroids, volumes = ff.getVolumes(str(sim_dir))
        print(f"  fluidfoam returned {len(centroids)} centroids")
        
        # Take only the first n_cells (these should be the cell centers)
        if len(centroids) >= n_cells:
            centroids = centroids[:n_cells]
        else:
            print(f"  Warning: fluidfoam returned fewer centroids than cells")
            
        x = centroids[:, 0]
        y = centroids[:, 1]
        z = centroids[:, 2] if centroids.shape[1] > 2 else np.zeros_like(x)
        
    except Exception as e:
        print(f"  Error reading mesh: {e}")
        return None
    
    # Read U and p fields
    try:
        Ux, Uy, Uz = ff.readvector(str(sim_dir), time_name="1000", name="U")
        p = ff.readscalar(str(sim_dir), time_name="1000", name="p")
        
        # Take only the first n_cells values
        u = np.asarray(Ux).ravel()[:n_cells]
        v = np.asarray(Uy).ravel()[:n_cells]
        w = np.asarray(Uz).ravel()[:n_cells]
        p = np.asarray(p).ravel()[:n_cells]
        
        print(f"  Fields shape: u={len(u)}, p={len(p)}")
        
    except Exception as e:
        print(f"  Error reading fields: {e}")
        return None
    
    # Check consistency
    if len(x) != len(u):
        print(f"  Length mismatch: x={len(x)}, u={len(u)}")
        # Try to take min of both
        min_len = min(len(x), len(u))
        x = x[:min_len]
        y = y[:min_len]
        z = z[:min_len]
        u = u[:min_len]
        v = v[:min_len]
        w = w[:min_len]
        p = p[:min_len]
        print(f"  Trimmed to {min_len} points")
    
    # Read parameters
    doe_path = sim_dir / "constant" / "DOE_row.txt"
    params = parse_doe_row(doe_path)
    params = add_dimensionless(params)
    
    # Create DataFrame
    df = pd.DataFrame({
        "sim": case_name,
        "x": x, "y": y, "z": z,
        "u": u, "v": v, "w": w, "p": p
    })
    
    for key, val in params.items():
        if isinstance(val, (int, float)):
            df[key] = val
    
    # Save
    csv_path = sim_dir / "cfd_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows to {csv_path}")
    
    return df


def process_root(root, label):
    print(f"\n=== Processing {label.upper()} simulations in: {root} ===")
    
    if not root.exists():
        print(f"Root not found: {root}")
        return
    
    sims = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("case_")])
    print(f"Found simulations:")
    for s in sims:
        print(f"  {s.name}")
    
    all_dfs = []
    for sim in sims:
        df = process_single_sim(sim)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print(f"[WARN] No data extracted from any {label} simulation.")
        return
    
    # Combine
    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(DATA_FOLDER, f"cfd_data_{label}_fluidfoam.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\n[OK] Combined CSV saved: {csv_path}")
    print(f"  Total rows: {len(combined)}")
    
    # PINN format
    X_cols = ["x", "y", "z"] + PARAM_COLS
    Y_cols = ["u", "v", "w", "p"]
    
    mask = combined[PARAM_COLS].notna().all(axis=1)
    pinn_df = combined[mask]
    
    X = pinn_df[X_cols].to_numpy(dtype=np.float32)
    Y = pinn_df[Y_cols].to_numpy(dtype=np.float32)
    
    npz_path = os.path.join(DATA_FOLDER, f"cfd_data_pinn_{label}.npz")
    np.savez(npz_path, X=X, Y=Y)
    print(f"[OK] PINN data saved: {npz_path}")
    print(f"{label.upper()} X shape: {X.shape}, Y shape: {Y.shape}")


def main():
    process_root(TRAIN_ROOT, "train")
    process_root(VAL_ROOT, "val")
    print(f"\n[SUCCESS] All files saved to: {DATA_FOLDER}/")


if __name__ == "__main__":
    main()
