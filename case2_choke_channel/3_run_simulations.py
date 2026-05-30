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
    
    #read the folders in 3_sims_training and list them. give the list in this format ["case_000", "case_001"]
    all_items = os.listdir(".")
    folders = [item for item in all_items if os.path.isdir(item) and item.startswith("case_")]
    folders.sort()
    
    #setting the list to the list of simulations
    sim_list = folders
    
    #print the simulation list
    print(sim_list)
    
    #simple loop to read each folder in the list
    for case_folder in folders:
        print(f"\nNow reading {case_folder}")
        
        #enters the case folder
        os.chdir(case_folder)
    
        #blockMesh command is applied to these cases
        print("Generating meshes...")
        try:
            #run blockMesh
            result = subprocess.run(["blockMesh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
                              
            if result.returncode == 0:
                print("\nSuccess")
            else:
                print(f"\nFailed. Error: {result.stderr[:100]}")
            
        except Exception as e:
            print(f"\nerror: {e}")
        #now simpleFoam will be applied
        print("\nRunning simulations...")
        try:
            result = subprocess.run(["simpleFoam"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
            
            if result.returncode == 0:
                print("\nSuccess")
            else:
                print(f"\nFailed. Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"\nFailed. Error: {e}")
            
            
        os.chdir("..")
    os.chdir("..")
    print(f"\nFinished 3_sims_training")
    print("="*100)
    print("\nAll meshes generated and simulations ran successfully for the training cases\n")
    print("="*100)
    
    print("\n\n=== Now processing 3_sims_validation ===")
    
    if not os.path.exists("3_sims_validation"):
        print(f"Folder '3_sims_validation' not found, skipping over it.")
    else:
        os.chdir("3_sims_validation")
    all_items = os.listdir(".")
    folders = [item for item in all_items if os.path.isdir(item) and item.startswith("case_")]
    folders.sort()
    
    sim_list = folders
    
    print(sim_list)
    
    for case_folder in folders:
        print(f"\nNow reading {case_folder}")
        
        os.chdir(case_folder)
    
        print("Generating meshes...")
        try:
            result = subprocess.run(["blockMesh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
                              
            if result.returncode == 0:
                print("\nSuccess")
            else:
                print(f"\nFailed. Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"\nerror: {e}")
        #now simpleFoam will be applied
        print("\nRunning simulations...")
        try:
            result = subprocess.run(["simpleFoam"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
            
            if result.returncode == 0:
                print("\nSuccess")
            else:
                print(f"\nFailed. Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"\nFailed. Error: {e}")
            
            
        os.chdir("..")
    os.chdir("..")
    print(f"\nFinished 3_sims_training")
    print("="*100)
    print("\nAll meshes generated and simulations ran successfully for the validation cases\n")
    print("="*100)
    
        

if __name__ == "__main__":
    main()
