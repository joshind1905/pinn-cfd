#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil
import random
import os
import subprocess

# ============================================================
# FIXED PATHS
# ============================================================
TRAIN_ROOT    = Path("3_sims_training")
VAL_ROOT      = Path("3_sims_validation")

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    
    print(f"\n=== Processing 3_sims_training ===")
        
    if not os.path.exists("3_sims_training"):
        print(f"Folder '3_sims_training' not found, skipping over it")
    else:
        #enter the folder 3_sims_training
        os.chdir("3_sims_training")
    
        #read the folders in 3_sims_training and list them
        all_items = os.listdir(".")
        folders = [item for item in all_items if os.path.isdir(item) and item.startswith("case_")]
        folders.sort()
        
        #print the simulation list
        print(folders)
        
        #simple loop to read each folder in the list
        for case_folder in folders:
            print(f"\nNow reading {case_folder}")
            
            #enters the case folder
            os.chdir(case_folder)
            
            # Check if mesh exists
            if not os.path.exists("constant/polyMesh"):
                print("  Mesh not found! Skipping...")
                os.chdir("..")
                continue
            
            # Run simpleFoam
            print("  Running simulation...")
            try:
                result = subprocess.run(["simpleFoam"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
                                  
                if result.returncode == 0:
                    print("  Success")
                else:
                    print(f"  Failed. Error: {result.stderr[:200]}")
                
            except Exception as e:
                print(f"  error: {e}")
                
            os.chdir("..")
        os.chdir("..")
    
    print(f"\nFinished 3_sims_training")
    print("="*100)
    
    print("\n\n=== Now processing 3_sims_validation ===")
    
    if not os.path.exists("3_sims_validation"):
        print(f"Folder '3_sims_validation' not found, skipping over it.")
    else:
        os.chdir("3_sims_validation")
        
        all_items = os.listdir(".")
        folders = [item for item in all_items if os.path.isdir(item) and item.startswith("case_")]
        folders.sort()
        
        print(folders)
        
        for case_folder in folders:
            print(f"\nNow reading {case_folder}")
            
            os.chdir(case_folder)
            
            # Check if mesh exists
            if not os.path.exists("constant/polyMesh"):
                print("  Mesh not found! Skipping...")
                os.chdir("..")
                continue
            
            print("  Running simulation...")
            try:
                result = subprocess.run(["simpleFoam"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
                
                if result.returncode == 0:
                    print("  Success")
                else:
                    print(f"  Failed. Error: {result.stderr[:200]}")
            except Exception as e:
                print(f"  error: {e}")
                
            os.chdir("..")
        os.chdir("..")
    
    print(f"\nFinished 3_sims_validation")
    print("="*100)
    print("\nAll simulations completed")
    print("="*100)

if __name__ == "__main__":
    main()
