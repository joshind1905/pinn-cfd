#!/usr/bin/env python3
"""
2_create_sims.py - Using Python Gmsh API
"""

import csv
import shutil
import random
import os
import subprocess
import time
from pathlib import Path
import gmsh


# ============================================================
# USER SETTINGS
# ============================================================
val_frac = 0.20
random_seed = 42


# ============================================================
# FIXED PATHS
# ============================================================
TEMPLATE_CASE = Path("template_onepillar")
DOE_FILE = Path("data_files") / "doe_scaled.csv"
TRAIN_ROOT = Path("3_sims_training")
VAL_ROOT = Path("3_sims_validation")

# FIXED MESH SIZE - same for all cases so geometry shouldnt have a large range, ensures fast meshing
FIXED_MESH_SIZE = 0.075  # adjust for mesh fineness (lower value) /coarseness (higher)


def generate_mesh(case_path, params):
    """Generate mesh using Python Gmsh API with FIXED coarse mesh size"""
    
    channel_length = params.get("channel_length", 1.0)
    channel_width = params.get("channel_width", 0.3)
    channel_height = params.get("channel_height", 0.3)
    pillar_radius = params.get("pillar_radius", 0.05)
    
    print(f"    Mesh: L={channel_length:.3f}, W={channel_width:.3f}, H={channel_height:.3f}, R={pillar_radius:.3f}")
    print(f"    FIXED mesh size: {FIXED_MESH_SIZE:.4f} ")
    
    try:
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.model.add("ChannelGeometry")
        
        # Create the channel geometry
        x_start = 0.0
        y_start = -channel_width / 2
        z_start = -channel_height / 2
        channel_volume = gmsh.model.occ.addBox(
            x_start, y_start, z_start,
            channel_length, channel_width, channel_height
        )
        
        # Create pillar
        pillar_x = channel_length / 2
        pillar_y = 0.0
        pillar_z = z_start
        pillar = gmsh.model.occ.addCylinder(
            pillar_x, pillar_y, pillar_z,
            0, 0, channel_height,
            pillar_radius
        )
        
        gmsh.model.occ.synchronize()
        
        # Subtract pillar from channel
        channel_volume = gmsh.model.occ.cut([(3, channel_volume)], [(3, pillar)])
        gmsh.model.occ.synchronize()
        
        # Get the resulting volume
        all_volumes = gmsh.model.getEntities(dim=3)
        channel_volume = all_volumes[0][1]
        
        # Set mesh size - FIXED coarse value
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), FIXED_MESH_SIZE)
        
        # Tag surfaces for boundaries
        surfaces = gmsh.model.getEntities(dim=2)
        
        inlet_surface = None
        outlet_surface = None
        wall_surfaces = []
        pillar_surfaces = []
        
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            
            # Inlet (X ≈ 0)
            if abs(com[0] - x_start) < 1e-6:
                inlet_surface = surface[1]
            # Outlet (X ≈ channel_length)
            elif abs(com[0] - (x_start + channel_length)) < 1e-6:
                outlet_surface = surface[1]
            # Pillar (within center region)
            elif (pillar_x - pillar_radius <= com[0] <= pillar_x + pillar_radius and 
                  pillar_y - pillar_radius <= com[1] <= pillar_y + pillar_radius):
                pillar_surfaces.append(surface[1])
            # Walls (everything else)
            else:
                wall_surfaces.append(surface[1])
        
        # Assign physical groups
        if inlet_surface:
            gmsh.model.addPhysicalGroup(2, [inlet_surface], tag=1)
            gmsh.model.setPhysicalName(2, 1, "inlet")
        if outlet_surface:
            gmsh.model.addPhysicalGroup(2, [outlet_surface], tag=2)
            gmsh.model.setPhysicalName(2, 2, "outlet")
        
        # Add walls and pillar as separate groups
        gmsh.model.addPhysicalGroup(2, wall_surfaces, tag=3)
        gmsh.model.setPhysicalName(2, 3, "walls")
        
        if pillar_surfaces:
            gmsh.model.addPhysicalGroup(2, pillar_surfaces, tag=4)
            gmsh.model.setPhysicalName(2, 4, "pillar")
        
        # Add volume
        gmsh.model.addPhysicalGroup(3, [channel_volume], tag=5)
        gmsh.model.setPhysicalName(3, 5, "channelVolume")
        
        # Set mesh format and generate
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        
        # Save mesh
        mesh_file = case_path / "channel_geometry.msh"
        gmsh.write(str(mesh_file))
        
        # Finalize
        gmsh.finalize()
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        try:
            gmsh.finalize()
        except:
            pass
        return False


def build_case_name(idx: int) -> str:
    return f"case_{idx+1:03d}"


def write_doe_row_txt(path: Path, row: dict):
    lines = ["// Simulation parameters for ONEPILLAR case\n", "// Generated by 2_create_sims.py\n\n"]
    for key, value in row.items():
        v = str(value).strip()
        if v and v.lower() != "nan":
            lines.append(f"{key:<20} {v};\n")
    path.write_text("".join(lines))


def setup_openfoam_case(case_path, params):
    """Set up OpenFOAM files with parameters"""
    
    # Replace U_ave in the U file
    u_file = case_path / "0" / "U"
    if u_file.exists() and "U_ave" in params:
        content = u_file.read_text()
        import re
        new_content = re.sub(r'value\s+uniform\s+\([^)]+\)', 
                             f'value           uniform ({params["U_ave"]} 0 0)', 
                             content)
        u_file.write_text(new_content)
        print(f"    Updated U_ave to {params['U_ave']}")

    # Replace nu in transportProperties
    transport_file = case_path / "constant" / "transportProperties"
    if transport_file.exists() and "nu" in params:
        content = transport_file.read_text()
        import re
        new_content = re.sub(r'nu\s+nu\s+\[[^\]]+\]\s+[0-9eE.-]+', 
                             f'nu              nu [0 2 -1 0 0 0 0] {params["nu"]}', 
                             content)
        transport_file.write_text(new_content)
        print(f"    Updated nu to {params['nu']}")


def find_gmshtofoam():
    possible_paths = [
        "/home/chrisj/OpenFOAM/OpenFOAM-5.x/bin/gmshToFoam",
        "/opt/openfoam5/bin/gmshToFoam",
        "/usr/lib/openfoam/openfoam5/bin/gmshToFoam",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    try:
        result = subprocess.run(["which", "gmshToFoam"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "gmshToFoam"


def main():
    print("="*60)
    print("ONEPILLAR CASE CREATION")
    print("="*60)
    print(f"Validation fraction = {val_frac*100:.1f}%")
    print(f"FIXED mesh size = {FIXED_MESH_SIZE:.4f} ")
    random.seed(random_seed)

    if not TEMPLATE_CASE.is_dir():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_CASE}")
    if not DOE_FILE.is_file():
        raise FileNotFoundError(f"DOE file not found: {DOE_FILE}")

    gmshToFoam_cmd = find_gmshtofoam()
    print(f"Using gmshToFoam: {gmshToFoam_cmd}")

    rows = []
    with DOE_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            for key in row:
                try:
                    row[key] = float(row[key])
                except:
                    pass
            rows.append((idx, row))

    n_sims = len(rows)
    print(f"\nFound {n_sims} simulations")

    # Remove old case folders
    for folder in [TRAIN_ROOT, VAL_ROOT]:
        if folder.exists():
            print(f"Removing old {folder}...")
            shutil.rmtree(folder)

    TRAIN_ROOT.mkdir(exist_ok=True)
    VAL_ROOT.mkdir(exist_ok=True)

    indices = list(range(n_sims))
    random.shuffle(indices)
    n_val = max(1, int(round(val_frac * n_sims)))
    n_val = min(n_val, n_sims - 1)
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    print(f"Training: {len(train_indices)} | Validation: {len(val_indices)}")
    print(f"\nCreating cases...")

    train_count = val_count = mesh_success = mesh_failed = 0

    for idx, row in rows:
        if idx in train_indices:
            root, label = TRAIN_ROOT, "TRAIN"
            train_count += 1
        else:
            root, label = VAL_ROOT, "VAL"
            val_count += 1

        case_name = build_case_name(idx)
        dest = root / case_name

        print(f"\n[CREATE] ({label}) {case_name}")
        shutil.copytree(TEMPLATE_CASE, dest)
        
        # Write DOE_row.txt to the constant directory
        write_doe_row_txt(dest / "constant" / "DOE_row.txt", row)
        setup_openfoam_case(dest, row)

        print(f"  Generating mesh...")
        if generate_mesh(dest, row):
            mesh_success += 1
            print(f"    Mesh generation complete")
            print(f"  Converting to OpenFOAM...")
            try:
                os.chdir(dest)
                result = subprocess.run([gmshToFoam_cmd, "channel_geometry.msh"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True, timeout=120)
                if result.returncode == 0:
                    print(f"    Conversion OK")
                else:
                    print(f"    Conversion failed: {result.stderr[:300]}")
                    mesh_failed += 1
                os.chdir("../..")
            except Exception as e:
                print(f"    Conversion error: {e}")
                mesh_failed += 1
                os.chdir("../..")
        else:
            mesh_failed += 1

    print(f"\n{'='*60}")
    print(f"COMPLETE: {train_count} training, {val_count} validation")
    print(f"Mesh success: {mesh_success} | Failed: {mesh_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
