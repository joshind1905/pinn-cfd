#!/usr/bin/env python3
"""
create_sims_train_val.py

This script reads 2_DOE/doe_scaled.csv, splits the simulations into a training
set and a validation set according to val_frac, and creates two folders:

    3_sims_training/
    3_sims_validation/

Each contains folders like sim_1/, sim_2/, ... copied from 1_baseChoke2D/.
In each case folder a DOE_row.txt file is written with the parameters for OpenFOAM.
"""

from pathlib import Path
import csv
import shutil
import random


# ============================================================
# USER SETTING: VALIDATION FRACTION
# ============================================================
val_frac = 0.20      # <-- fraction of simulations to send to validation
random_seed = 42     # reproducible shuffle


# ============================================================
# FIXED PATHS
# ============================================================
TEMPLATE_CASE = Path("1_baseChoke2D")
DOE_FILE      = Path("2_DOE") / "doe_scaled.csv"
TRAIN_ROOT    = Path("3_sims_training")
VAL_ROOT      = Path("3_sims_validation")

#Fixed geometry parameters

FIXED_GEOMETRY = {
    'H': 500.0,      # Total height (μm)
    'L': 2000.0,     # Total length (μm)
    'h': 50.0,       # Constriction height (μm)
    'l': 200.0,      # Constriction length (μm)
}
# ============================================================
# SUPPORT FUNCTIONS (same logic as your original script)
# ============================================================

def build_case_name(row: dict, idx: int) -> str:
    """Build case folder name sim_<id> or fallback case_000."""
    if "sim" in row:
        raw = str(row["sim"]).strip()
        if raw != "":
            try:
                sim_int = int(float(raw))
                sim_value = str(sim_int)
            except ValueError:
                sim_value = raw
            return f"sim_{sim_value}"
    return f"case_{idx:03d}"


def add_derived_geometry(row: dict) -> dict:
    """Compute x1, x2, y1, y2 from H, L, h, l (if available)."""
    out = dict(row)
    try:
        H = float(row["H"])
        L = float(row["L"])
        h = float(row["h"])
        l = float(row["l"])
    except Exception:
        return out

    out["x1"] = 0.5 * (L - l)
    out["x2"] = 0.5 * (L + l)
    out["y1"] = 0.5 * (H - h)
    out["y2"] = out["y1"] + h
    return out


def add_physical_params(row: dict) -> dict:
    """Ensure U_ave and kin_vis fields exist."""
    out = dict(row)

    if "U_ave" not in out or str(out["U_ave"]).strip() == "":
        for alt in ("Uave", "U_mean", "Umean"):
            if alt in row and str(row[alt]).strip() != "":
                out["U_ave"] = row[alt]

    if "kin_vis" not in out or str(out["kin_vis"]).strip() == "":
        for alt in ("kinematic_viscosity", "nu", "nu_eff"):
            if alt in row and str(row[alt]).strip() != "":
                out["kin_vis"] = row[alt]

    return out


def write_doe_row_txt(path: Path, row: dict):
    """Write OpenFOAM-style key–value pairs."""
    lines = []
    for key, value in row.items():
        v = str(value).strip()
        if v != "":
            lines.append(f"{key}    {v};\n")
    path.write_text("".join(lines))


# ============================================================
# MAIN LOGIC
# ============================================================

def main():

    print(f"Validation fraction = {val_frac*100:.1f}%")
    random.seed(random_seed)

    # Check template case
    if not TEMPLATE_CASE.is_dir():
        raise FileNotFoundError(f"Template case not found: {TEMPLATE_CASE}")

    # Check DOE file
    if not DOE_FILE.is_file():
        raise FileNotFoundError(f"DOE file not found: {DOE_FILE}")

    # Read DOE rows
    rows = []
    with DOE_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append((idx, row))

    if len(rows) == 0:
        raise ValueError("DOE CSV contains no rows.")

    n_sims = len(rows)
    print(f"Found {n_sims} simulations in DOE.")

    # Create output folders
    TRAIN_ROOT.mkdir(exist_ok=True)
    VAL_ROOT.mkdir(exist_ok=True)

    # Randomly split simulations
    indices = list(range(n_sims))
    random.shuffle(indices)

    n_val = max(1, int(round(val_frac * n_sims)))
    n_val = min(n_val, n_sims - 1)

    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    print(f"Training sims:  {len(train_indices)}")
    print(f"Validation sims:{len(val_indices)}")

    # Process each simulation
    for idx, row in rows:
        # Determine target folder
        if idx in train_indices:
            root = TRAIN_ROOT
            label = "TRAIN"
        else:
            root = VAL_ROOT
            label = "VAL"

        # Build case name
        case_name = build_case_name(row, idx)
        dest = root / case_name

        if dest.exists():
            print(f"[SKIP] ({label}) {dest} already exists.")
            continue

        # Create full row with geometry + parameters
        row_full = add_physical_params(add_derived_geometry(row))

        print(f"[CREATE] ({label}) {dest}  ←  {TEMPLATE_CASE}")
        shutil.copytree(TEMPLATE_CASE, dest)

        # Write DOE_row.txt
        write_doe_row_txt(dest / "DOE_row.txt", row_full)


if __name__ == "__main__":
    main()
