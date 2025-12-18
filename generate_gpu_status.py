import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
import glob
from PIL import Image
import sys
import re

# =================CONFIGURATION=================
# Comma-separated list of your GPU partition names
PARTITIONS = "gpu,gpu-test,gpu-contrib,gpu-general"

# Directory to store daily images
ARCHIVE_DIR = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_archive"

# Filename for the combined historical GIF
GIF_FILENAME = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_history.gif"

# Command timeout in seconds
TIMEOUT = 15

# === HARDWARE TRUTH MAP ===
# Defines what *should* be there. 
# Format: (Node_Pattern, GPU_Model_Name, Expected_Count)
HARDWARE_DEFINITIONS = [
    ("g20-[01-04]", "RTX 2080Ti", 8),
    ("g20-[05-11]", "RTX 6000", 8),
    ("g20-[12-13]", "RTX 8000", 8),
    ("g24-[09-10]", "H100", 2),
    ("g24-[01-08,11-12]", "L40S", 4)
]
# ===============================================

def run_command(command):
    """Executes a shell command and returns stdout, stderr, and return code."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 124

def expand_hostlist(hostlist_expr):
    """Uses scontrol to expand bracket notation (e.g. node[01-03] -> node01, node02...)"""
    if not hostlist_expr: return []
    # Clean up any spaces in the expression
    hostlist_expr = hostlist_expr.replace(" ", "")
    cmd = f"scontrol show hostnames '{hostlist_expr}'"
    stdout, stderr, rc = run_command(cmd)
    if rc == 0:
        return stdout.splitlines()
    return [hostlist_expr]

def build_hardware_map():
    """Compiles the HARDWARE_DEFINITIONS into a fast lookup dict: {node_name: (model, count)}"""
    mapping = {}
    print("Building Hardware Truth Map...")
    for pattern, model, count in HARDWARE_DEFINITIONS:
        nodes = expand_hostlist(pattern)
        for node in nodes:
            mapping[node] = (model, count)
    return mapping

def get_gpu_data(hardware_map):
    data = {}
    print("Fetching node list from SLURM...")
    
    # Get list of nodes in partition
    sinfo_cmd = f"sinfo -p {PARTITIONS} -h -o '%N|%t'"
    stdout, stderr, rc = run_command(sinfo_cmd)
    if rc != 0:
        print(f"Error running sinfo: {stderr}")
        sys.exit(1)

    lines = stdout.split('\n')
    
    # Process nodes
    processed_nodes = set()
    
    for line in lines:
        if not line.strip(): continue
        parts = line.split('|')
        if len(parts) < 2: continue
        
        compressed_nodes, node_state = parts[0], parts[1]
        individual_nodes = expand_hostlist(compressed_nodes)
        
        for node_name in individual_nodes:
            if node_name in processed_nodes: continue
            processed_nodes.add(node_name)
            
            # Default to checking our Truth Map first
            if node_name in hardware_map:
                expected_model, expected_count = hardware_map[node_name]
            else:
                # Fallback if a new node appears that isn't in our config
                expected_model, expected_count = ("Unknown New GPU", 0)

            data[node_name] = {
                "slurm_state": node_state,
                "gpus": []
            }

            # Check accessibility
            is_accessible = True
            if any(x in node_state.lower() for x in ["down", "drng", "drain", "*"]):
                is_accessible = False

            smi_success = False
            
            # === STRATEGY 1: QUERY (If accessible) ===
            if is_accessible:
                # Added --pty to potentially help with shell allocation, and explicitly specific partition if needed
                # Also capture stderr to debug why it's failing
                nvidia_cmd = f"srun -w {node_name} -n1 -N1 --exclusive --gpu-bind=none nvidia-smi --query-gpu=index --format=csv,noheader"
                gpu_stdout, gpu_stderr, gpu_rc = run_command(nvidia_cmd)

                if gpu_rc == 0 and gpu_stdout:
                    smi_success = True
                    found_indices = [x.strip() for x in gpu_stdout.splitlines() if x.strip()]
                    
                    # 1. Add Found GPUs (Green)
                    for idx in found_indices:
                        data[node_name]["gpus"].append({
                            "model": expected_model,
                            "index": idx,
                            "state": "good"
                        })
                    
                    # 2. Add Missing GPUs (Red)
                    # If we expected 8 but found 5, we need to add 3 red blocks
                    if len(found_indices) < expected_count:
                        # Find which indices are missing (assuming 0-indexed)
                        found_set = set(map(int, found_indices))
                        for i in range(expected_count):
                            if i not in found_set:
                                data[node_name]["gpus"].append({
                                    "model": expected_model,
                                    "index": str(i),
                                    "state": "bad" # Explicitly missing
                                })
                else:
                    # Print why srun failed to help debugging
                    print(f"  > Warning: srun failed for {node_name} ({node_state}).") 
                    # Uncomment next line to see specific error in logs:
                    # print(f"    Error: {gpu_stderr}")

            # === STRATEGY 2: FILL FROM MAP (If query failed) ===
            if not smi_success:
                # If we couldn't query, we assume ALL expected GPUs are there but in Bad state
                if expected_count > 0:
                    for i in range(expected_count):
                        data[node_name]["gpus"].append({
                            "model": expected_model,
                            "index": str(i),
                            "state": "bad"
                        })

            # Sort GPUs by index so diagram looks neat
            data[node_name]["gpus"].sort(key=lambda x: int(x['index']) if x['index'].isdigit() else 999)

    return data

def organize_data_by_type(raw_data):
    """Reorganizes data from node-centric to GPU-type-centric."""
    organized = {}
    for node_name, node_info in raw_data.items():
        if not node_info["gpus"]: continue
        
        # Group by the MODEL name we assigned
        model = node_info["gpus"][0]["model"]
        
        if model not in organized:
            organized[model] = {}
        if node_name not in organized[model]:
            organized[model][node_name] = []
        organized[model][node_name].extend(node_info["gpus"])
        
    return dict(sorted(organized.items()))

def create_status_image(data, filename):
    if not data: return

    # Calculate dynamic height
    total_rows = 0
    for model in data:
        n_nodes = len(data[model])
        rows = (n_nodes // 5) + 2
        total_rows += rows
    fig_height = max(8, total_rows * 1.8)
    
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_axis_off()
    
    # Header
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.98, "GPU Infrastructure Status Diagram", ha='center', va='top', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.96, f"Last Updated: {current_date}", ha='center', va='top', fontsize=16, color='#555')

    # Legend
    legend_y = 0.96
    ax.add_patch(patches.Rectangle((0.82, legend_y), 0.02, 0.015, facecolor='#28a745', edgecolor='black'))
    ax.text(0.85, legend_y, "Operational", va='bottom', fontsize=12)
    ax.add_patch(patches.Rectangle((0.82, legend_y - 0.02), 0.02, 0.015, facecolor='#dc3545', edgecolor='black'))
    ax.text(0.85, legend_y - 0.02, "Down / Missing", va='bottom', fontsize=12)

    y_cursor = 0.92
    SERVER_WIDTH = 0.15
    SERVER_HEIGHT = 0.08
    X_START = 0.05
    X_GAP = 0.02
    Y_GAP = 0.06
    
    for gpu_type, nodes in data.items():
        # Section Header
        ax.text(X_START, y_cursor, gpu_type, fontsize=18, fontweight='bold')
        ax.add_patch(patches.Rectangle((X_START, y_cursor - 0.005), 0.9, 0.002, color='black'))
        y_cursor -= 0.05
        
        x_cursor = X_START
        sorted_nodes = sorted(nodes.keys())
        
        for node_name in sorted_nodes:
            gpus = nodes[node_name]
            
            # Wrap to new line
            if x_cursor + SERVER_WIDTH > 0.98:
                x_cursor = X_START
                y_cursor -= (SERVER_HEIGHT + 0.04)

            # Server Box
            ax.add_patch(patches.Rectangle((x_cursor, y_cursor - SERVER_HEIGHT), SERVER_WIDTH, SERVER_HEIGHT, 
                                           facecolor='#f8f9fa', edgecolor='#6c757d', linewidth=1.5))
            ax.text(x_cursor + 0.005, y_cursor - 0.015, node_name, fontsize=11, fontweight='bold')
            
            # GPU Boxes
            # Dynamic sizing: if 8 GPUs, 4 per row. If 2 GPUs, 2 per row.
            total_gpus = len(gpus)
            cols = 4 if total_gpus > 4 else total_gpus
            cols = max(1, cols) # avoid div by zero
            
            gpu_w = (SERVER_WIDTH - 0.02) / cols 
            gpu_h = 0.02
            
            gx_start = x_cursor + 0.01
            gy_start = y_cursor - 0.035
            gx, gy = gx_start, gy_start
            
            for i, gpu in enumerate(gpus):
                color = '#28a745' if gpu['state'] == 'good' else '#dc3545'
                rect = patches.Rectangle((gx, gy - gpu_h), gpu_w - 0.002, gpu_h, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                # Index Label
                ax.text(gx + (gpu_w/2), gy - (gpu_h/2), gpu['index'], 
                        ha='center', va='center', color='white', fontsize=7, fontweight='bold')
                
                gx += gpu_w
                # Wrap internal rows
                if (i + 1) % cols == 0:
                    gx = gx_start
                    gy -= (gpu_h + 0.005)

            x_cursor += (SERVER_WIDTH + X_GAP)
            
        y_cursor -= (SERVER_HEIGHT + Y_GAP)

    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Image generated: {filename}")

def create_gif(image_folder, gif_filename):
    filenames = sorted(glob.glob(os.path.join(image_folder, "gpu_status_*.png")))[-30:]
    if not filenames: return
    images = [Image.open(f) for f in filenames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=1000, loop=0)
    print(f"GIF updated: {gif_filename}")

if __name__ == "__main__":
    print("--- Starting GPU Status Generation ---")
    if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 1. Build Truth Map
    hw_map = build_hardware_map()
    
    # 2. Gather Data (comparing Truth vs Reality)
    data = get_gpu_data(hw_map)
    organized = organize_data_by_type(data)
    
    # 3. Generate Artifacts
    today = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(ARCHIVE_DIR, f"gpu_status_{today}.png")
    create_status_image(organized, out_file)
    create_gif(ARCHIVE_DIR, GIF_FILENAME)
    print("--- Finished ---")
