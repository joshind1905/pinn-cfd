#!/usr/bin/env python3
"""
2_postprocess.py

Post-process all OpenFOAM cases in:

    3_sims_training/sim_*
    3_sims_validation/sim_*

using fluidfoam to build two datasets:

    - cfd_data_train_fluidfoam.csv
    - cfd_data_val_fluidfoam.csv

For each case:
  - read cell centroids with fluidfoam.getVolumes(path)  [constant/polyMesh]
  - read vector field U and scalar field p with fluidfoam.readvector/readscalar
    at time_name='latestTime'
  - extract x, y, u, v, p
  - read DOE_row.txt for parameters (U_ave, kin_vis, H, L, h, l, x1, x2, y1, y2, ...)
  - compute Re and beta
  - build a DataFrame and (optionally) a per-case CSV

Finally:
  - one CSV per split (train / val)
  - one NPZ per split with X, Y arrays for PINN.

Run from the PINN project root:

    (pinn-cfd) python3 2_postprocess.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import fluidfoam as ff
import os  # Added import


# Roots for training and validation simulations
TRAIN_ROOT = Path("3_sims_training").resolve()
VAL_ROOT   = Path("3_sims_validation").resolve()

# Also write sim_i/cfd_data_fluidfoam.csv inside each case?
SAVE_PER_SIM = True

#define parameters
PARAM_COLS = ['U_ave', 'kin_vis']

# ------------------------------------------------------------------
# ADDED: Create data_files folder for all output
# ------------------------------------------------------------------
DATA_FOLDER = "data_files"
os.makedirs(DATA_FOLDER, exist_ok=True)
print(f"[INFO] All output files will be saved to: {DATA_FOLDER}")
# ------------------------------------------------------------------

# ---------- Utilities: read DOE_row.txt ----------

def parse_doe_row(doe_path: Path) -> dict:
    """
    Parse DOE_row.txt with lines like:
        key    value;
    or      key = value;

    Returns a dict {key: float_or_str}.
    """
    params = {}
    if not doe_path.is_file():
        print(f"[WARN] DOE_row.txt not found: {doe_path}")
        return params

    for line in doe_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        line = line.replace("=", " ")
        line = line.replace(";", " ")
        tokens = line.split()
        if len(tokens) >= 2:
            key = tokens[0]
            val_str = tokens[1]
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            params[key] = val
    return params


def add_dimensionless(params: dict) -> dict:
    """
    Compute and add Re and beta if possible.

    Assumes:
      H, h in microns
      U_ave in m/s
      kin_vis in m2/s

    Re_H = U_ave * H[m] / nu
    beta = h / H
    """
    out = dict(params)

    H_um = out.get("H", None)
    h_um = out.get("h", None)
    U_ave = out.get("U_ave", None)
    nu = out.get("kin_vis", None)
    
    # ------------------------------------------------------------------
    # FIXED: Convert parameters to float before using them
    # ------------------------------------------------------------------
    try:
        if H_um is not None:
            H_um = float(H_um)
        if h_um is not None:
            h_um = float(h_um)
        if U_ave is not None:
            U_ave = float(U_ave)
        if nu is not None:
            nu = float(nu)
    except (ValueError, TypeError) as e:
        # If conversion fails, return params without dimensionless numbers
        print(f"[WARN] Could not convert parameters to float in add_dimensionless: {e}")
        return out
    # ------------------------------------------------------------------

    if None not in (H_um, U_ave, nu):
        H_m = H_um * 1e-6
        Re = U_ave * H_m / nu
        out["Re"] = Re

    if None not in (H_um, h_um):
        beta = h_um / H_um
        out["beta"] = beta

    return out


# ---------- Process a single sim with fluidfoam ----------

def process_single_sim(sim_dir: Path) -> pd.DataFrame | None:
    """
    Read OpenFOAM case in sim_dir with fluidfoam and return DataFrame with:
      x, y, u, v, p + DOE parameters.
    """
    case_name = sim_dir.name
    doe_path = sim_dir / "DOE_row.txt"
    case_path = sim_dir

    print(f"[INFO] {case_name}: processing latestTime with fluidfoam")

    # 1) Read cell centroids and volumes from constant/polyMesh
    try:
        centroids, volumes = ff.getVolumes(str(case_path))  # no time_name -> constant/polyMesh
    except Exception as e:
        print(f"[WARN] {case_name}: failed to getVolumes: {e}")
        return None

    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.shape[1] < 2:
        print(f"[WARN] {case_name}: centroids have unexpected shape {centroids.shape}, skipping.")
        return None

    x = centroids[:, 0]
    y = centroids[:, 1]

    # 2) Read U and p at latestTime
    try:
        # readvector returns (Ux, Uy, Uz) arrays
        Ux, Uy, Uz = ff.readvector(str(case_path), time_name="latestTime", name="U")
        p = ff.readscalar(str(case_path), time_name="latestTime", name="p")
    except Exception as e:
        print(f"[WARN] {case_name}: failed to read fields U or p: {e}")
        return None

    u = np.asarray(Ux).ravel()
    v = np.asarray(Uy).ravel()
    p = np.asarray(p).ravel()

    # 3) Check sizes are consistent
    if not (len(x) == len(y) == len(u) == len(v) == len(p)):
        print(f"[WARN] {case_name}: inconsistent lengths x/y/u/v/p -> "
              f"{len(x)}, {len(y)}, {len(u)}, {len(v)}, {len(p)}; skipping.")
        return None

    # 4) Read DOE parameters and add Re, beta
    params = parse_doe_row(doe_path)
    params = add_dimensionless(params)

    # 5) Build DataFrame
    df = pd.DataFrame({
        "sim": case_name,
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "p": p,
    })

    # Attach parameters as extra columns
    for key, val in params.items():
        df[key] = val

    # Optional per-sim CSV
    if SAVE_PER_SIM:
        csv_path = sim_dir / "cfd_data_fluidfoam.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] {case_name}: per-sim CSV saved to {csv_path}")

    return df


# ---------- Helper: process all sims under a root ----------

def process_root(root: Path, label: str):
    """
    Process all sim_* directories under a given root and produce:
      - cfd_data_<label>_fluidfoam.csv
      - cfd_data_pinn_fluidfoam_<label>.npz
    """
    print(f"\n=== Processing {label.upper()} simulations in: {root} ===")

    if not root.is_dir():
        raise FileNotFoundError(f"Simulation root not found: {root}")

    # subdirectories (sim_*)
    sim_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not sim_dirs:
        raise FileNotFoundError(f"No sim_* directories found in {root}")

    print("Found simulations:")
    for d in sim_dirs:
        print("  ", d.name)

    all_dfs = []

    print("\nProcessing simulations with fluidfoam...\n")
    for sim_dir in sim_dirs:
        df = process_single_sim(sim_dir)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print(f"[WARN] No data extracted from any {label} simulation.")
        return

    big_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined CSV
    # ------------------------------------------------------------------
    # MODIFIED: Save to data_files folder
    # ------------------------------------------------------------------
    out_csv = os.path.join(DATA_FOLDER, f"cfd_data_{label}_fluidfoam.csv")
    # ------------------------------------------------------------------
    big_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Combined {label} CSV saved to: {out_csv}")

    # Build X,Y for PINN: example X = [x, y, Re, beta], Y = [u, v, p]
    #X_cols = ["x", "y", "Re", "beta"]
    X_cols = ["x", "y"] + PARAM_COLS
    Y_cols = ["u", "v", "p"]

    #if "Re" in big_df.columns and "beta" in big_df.columns:
    #    mask = big_df["Re"].notna() & big_df["beta"].notna()
    #    df_pinn = big_df[mask].copy()
    #else:
    #    print(f"[WARN] Re/beta not in {label} DataFrame, using all rows for X,Y.")
    #    df_pinn = big_df.copy()
     
    if "U_ave" in big_df.columns and "kin_vis" in big_df.columns:
        mask = big_df["U_ave"].notna() & big_df["kin_vis"].notna()
        df_pinn = big_df[mask].copy()
    else:
        print(f"[WARN] U_ave/kin_vis not in {label} DataFrame, using all rows for X,Y.")
        df_pinn = big_df.copy()


    X = df_pinn[X_cols].to_numpy(dtype=np.float32)
    Y = df_pinn[Y_cols].to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # MODIFIED: Save to data_files folder
    # ------------------------------------------------------------------
    npz_path = os.path.join(DATA_FOLDER, f"cfd_data_pinn_fluidfoam_{label}.npz")
    # ------------------------------------------------------------------
    np.savez(npz_path, X=X, Y=Y)
    print(f"[OK] {label} PINN data saved to: {npz_path}")
    print(f"{label.upper()} X shape:", X.shape, "Y shape:", Y.shape)


# ---------- Main ----------

def main():
    # Training split
    process_root(TRAIN_ROOT, label="train")

    # Validation split
    process_root(VAL_ROOT, label="val")


if __name__ == "__main__":
    main()
