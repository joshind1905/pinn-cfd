#!/usr/bin/env python3
"""
reset_experiment.py

Deletes all generated directories to reset the experiment.
"""

import os
import shutil

# Directories to remove
DIRS_TO_REMOVE = [
    "3_sims_training",
    "3_sims_validation",
    "data_files",
]

# Delete directories
for dir_name in DIRS_TO_REMOVE:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

print("✅ Reset complete")
