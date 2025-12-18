#!/bin/bash

# --- CONFIGURATION ---
# 1. Directory where your script lives
PROJECT_DIR="/umbc/rs/pi_doit/users/elliotg2/gpuGraphics" 

# 2. Python Executable
# OPTION A: The system python you just found (Use this if matplotlib is in base)
PYTHON_EXEC="/usr/ebuild/installs/software/Anaconda3/2024.02-1/bin/python"

# OPTION B: Your specific env (Use this if Option A fails with "Module not found")
# PYTHON_EXEC="/umbc/rs/pi_gobbert/users/elliotg2/conda_envs/BenchmarkingPythonEnvironment/bin/python"

# 3. Add Slurm binaries to the PATH (Crucial for scontrol)
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
# ---------------------

# Logging start time
echo "------------------------------------------------"
echo "Starting Scan: $(date)"

# Navigate to the project directory so images save in the right place
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# Run the script
"$PYTHON_EXEC" generate_gpu_status.py

echo "Scan Complete: $(date)"
