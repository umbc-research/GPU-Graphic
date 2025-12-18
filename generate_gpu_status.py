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
PARTITIONS = "gpu" 

# Directory to store daily images
ARCHIVE_DIR = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_archive"

# Filename for the combined historical GIF
GIF_FILENAME = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_history.gif"

# Command timeout in seconds (prevent hanging on down nodes)
TIMEOUT = 10
# ===============================================

def run_command(command):
    """Executes a shell command and returns stdout and return code."""
    try:
        # print(f"DEBUG Executing: {command}") 
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", 124 

def expand_hostlist(hostlist_expr):
    """
    Uses 'scontrol show hostnames' to expand SLURM bracket notation 
    (e.g., node[01-03]) into a python list of strings (node01, node02, node03).
    """
    if not hostlist_expr:
        return []
    # scontrol show hostnames takes the compressed list and outputs newline-separated hosts
    cmd = f"scontrol show hostnames '{hostlist_expr}'"
    stdout, rc = run_command(cmd)
    if rc == 0:
        return stdout.splitlines()
    else:
        # Fallback: if it's just a single node without brackets, return it
        return [hostlist_expr]

def clean_gpu_name(raw_name):
    """Maps raw nvidia-smi/slurm names to clean labels."""
    raw = raw_name.upper().replace("NVIDIA", "").replace("GEFORCE", "").strip()
    
    # Map for canonical names you want in the diagram
    mapping = {
        "RTX 2080 TI": "RTX 2080Ti",
        "2080TI": "RTX 2080Ti",
        "RTX 6000": "RTX 6000",
        "QUADRO RTX 6000": "RTX 6000",
        "RTX 8000": "RTX 8000",
        "QUADRO RTX 8000": "RTX 8000",
        "H100": "H100",
        "A100": "A100",
        "V100": "V100",
        "L40S": "L40S"
    }
    
    # Partial match check
    for key, val in mapping.items():
        if key in raw:
            return val
    
    return raw # Return cleaned raw string if no map found

def parse_slurm_gres(gres_str):
    """
    Parses SLURM GRES string to find GPU type and count.
    Standard formats: gpu:4, gpu:rtx2080ti:4, gpu:tesla:2
    Returns (model_name, count)
    """
    count = 0
    model = "Unknown GPU"
    
    # Check if this GRES string actually contains gpu info
    if "gpu" not in gres_str:
        return model, count

    try:
        # Split by comma first (gres can be 'gpu:2,mic:1')
        parts = gres_str.split(',')
        for part in parts:
            if part.startswith('gpu'):
                segments = part.split(':')
                # format: gpu:count OR gpu:type:count
                if len(segments) == 2:
                    # gpu:4
                    if segments[1].isdigit():
                        count = int(segments[1])
                elif len(segments) == 3:
                    # gpu:type:4
                    model = segments[1]
                    if segments[2].isdigit():
                        count = int(segments[2])
    except:
        pass
        
    return clean_gpu_name(model), count

def get_gpu_data():
    """
    Queries SLURM for nodes, expands lists, checks status.
    """
    data = {}
    print("Fetching node list from SLURM...")
    
    # Get Compressed Node List, State, and GRES
    # We use -r to try and prevent too much grouping, but scontrol expansion is safer
    sinfo_cmd = f"sinfo -p {PARTITIONS} -h -o '%N|%t|%G'"
    stdout, rc = run_command(sinfo_cmd)
    
    if rc != 0:
        print(f"Error running sinfo: {stdout}")
        sys.exit(1)

    lines = stdout.split('\n')
    print(f"Processing SLURM output lines...")

    for line in lines:
        if not line.strip(): continue
        parts = line.split('|')
        if len(parts) < 3: continue
        
        compressed_nodes, node_state, gres = parts[0], parts[1], parts[2]
        
        # 1. Expand the node list (e.g. g20-[01-05] -> g20-01, g20-02...)
        individual_nodes = expand_hostlist(compressed_nodes)
        
        for node_name in individual_nodes:
            data[node_name] = {
                "slurm_state": node_state,
                "gpus": []
            }
            
            # Identify if node is fundamentally accessible
            # "drng" = draining, "down" = down. We generally cannot SSH/srun to these.
            is_accessible = True
            if "down" in node_state.lower() or "drng" in node_state.lower() or "drain" in node_state.lower():
                is_accessible = False

            # === STRATEGY A: TRY NVIDIA-SMI (If accessible) ===
            smi_success = False
            if is_accessible:
                # print(f"  Querying {node_name}...")
                nvidia_cmd = f"srun -w {node_name} -n1 -N1 --exclusive --gpu-bind=none nvidia-smi --query-gpu=name,pci.bus_id,index --format=csv,noheader"
                gpu_stdout, gpu_rc = run_command(nvidia_cmd)

                if gpu_rc == 0 and gpu_stdout:
                    smi_success = True
                    for gpu_line in gpu_stdout.split('\n'):
                        if not gpu_line: continue
                        gpu_parts = gpu_line.split(',')
                        if len(gpu_parts) == 3:
                            raw_model, bus_id, index = [p.strip() for p in gpu_parts]
                            data[node_name]["gpus"].append({
                                "model": clean_gpu_name(raw_model),
                                "index": index,
                                "state": "good" # If we can query it and node isn't down, it's good
                            })

            # === STRATEGY B: FALLBACK TO SLURM GRES (If inaccessible or srun failed) ===
            if not smi_success:
                # Determine state: If node is explicitly Down/Drain, it's "bad". 
                # If it's Alloc/Mix/Idle but srun failed, it's also "bad" (unknown error).
                status_color = "bad"
                
                # Parse GRES to find out what *should* be there
                model_name, count = parse_slurm_gres(gres)
                
                # If parsed model is generic "Unknown", maybe map based on node name regex if you have naming conventions
                # e.g. if node starts with "g20", it's a 2080Ti. (Optional customization)
                
                if count > 0:
                    print(f"  Node {node_name} ({node_state}) inaccessible. Adding {count} x {model_name} as BAD.")
                    for i in range(count):
                        data[node_name]["gpus"].append({
                            "model": model_name,
                            "index": str(i),
                            "state": status_color
                        })
                else:
                    print(f"  Warning: Node {node_name} is down and has no GRES GPU info.")

    return data

def organize_data_by_type(raw_data):
    """Reorganizes data from node-centric to GPU-type-centric."""
    organized = {}
    for node_name, node_info in raw_data.items():
        if not node_info["gpus"]: continue
        
        # Assume all GPUs on a node are same type for grouping purposes
        # (Mixed GPU nodes are rare in HPC but possible)
        first_gpu = node_info["gpus"][0]
        model = first_gpu["model"]
        
        if model not in organized:
            organized[model] = {}
        
        # If node not present, add it
        if node_name not in organized[model]:
            organized[model][node_name] = []
            
        organized[model][node_name].extend(node_info["gpus"])
        
    return dict(sorted(organized.items()))

def create_status_image(data, filename):
    if not data:
        print("No data to plot.")
        return

    # Calculate height
    total_rows = 0
    for model in data:
        # crude estimation of rows needed per model
        n_nodes = len(data[model])
        rows = (n_nodes // 5) + 2
        total_rows += rows
    
    fig_height = max(8, total_rows * 1.5)
    
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_axis_off()
    
    # Header
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.98, "GPU Infrastructure Status Diagram", ha='center', va='top', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.96, f"Last Updated: {current_date}", ha='center', va='top', fontsize=16, color='#555')

    # Legend
    legend_y = 0.96
    ax.add_patch(patches.Rectangle((0.85, legend_y), 0.02, 0.015, facecolor='#28a745', edgecolor='black'))
    ax.text(0.88, legend_y, "Good State", va='bottom', fontsize=12)
    ax.add_patch(patches.Rectangle((0.85, legend_y - 0.02), 0.02, 0.015, facecolor='#dc3545', edgecolor='black'))
    ax.text(0.88, legend_y - 0.02, "Bad / Down", va='bottom', fontsize=12)

    # Layout Config
    y_cursor = 0.90
    SERVER_WIDTH = 0.15
    SERVER_HEIGHT = 0.08
    X_START = 0.05
    X_GAP = 0.02
    Y_GAP = 0.06
    
    for gpu_type, nodes in data.items():
        # Title for GPU Type
        ax.text(X_START, y_cursor, gpu_type, fontsize=18, fontweight='bold')
        ax.add_patch(patches.Rectangle((X_START, y_cursor - 0.005), 0.9, 0.002, color='black'))
        y_cursor -= 0.04
        
        x_cursor = X_START
        
        # Sort nodes naturally (g20-01, g20-02...)
        sorted_node_names = sorted(nodes.keys())
        
        for node_name in sorted_node_names:
            gpus = nodes[node_name]
            
            # Check for line wrap
            if x_cursor + SERVER_WIDTH > 0.98:
                x_cursor = X_START
                y_cursor -= (SERVER_HEIGHT + 0.04)

            # Draw Server Box
            # Determine if whole server is down (all GPUs bad) to maybe color the server box differently?
            # For now, keep server box grey.
            ax.add_patch(patches.Rectangle((x_cursor, y_cursor - SERVER_HEIGHT), SERVER_WIDTH, SERVER_HEIGHT, 
                                           facecolor='#f8f9fa', edgecolor='#6c757d', linewidth=1.5))
            
            # Server Label
            ax.text(x_cursor + 0.005, y_cursor - 0.015, node_name, fontsize=11, fontweight='bold')
            
            # Draw GPUs
            # We fit them inside the box.
            gpu_w = (SERVER_WIDTH - 0.02) / 4 # Assume max 4 per row inside box
            gpu_h = 0.02
            
            gx_start = x_cursor + 0.01
            gy_start = y_cursor - 0.035
            
            gx = gx_start
            gy = gy_start
            
            for i, gpu in enumerate(gpus):
                color = '#28a745' if gpu['state'] == 'good' else '#dc3545'
                
                rect = patches.Rectangle((gx, gy - gpu_h), gpu_w - 0.002, gpu_h, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                # Index label
                ax.text(gx + (gpu_w/2), gy - (gpu_h/2), gpu['index'], 
                        ha='center', va='center', color='white', fontsize=7, fontweight='bold')
                
                gx += gpu_w
                if (i + 1) % 4 == 0: # Wrap inside server box if > 4 gpus
                    gx = gx_start
                    gy -= (gpu_h + 0.005)

            x_cursor += (SERVER_WIDTH + X_GAP)
            
        # Move down for next GPU type section
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

# =================MAIN=================
if __name__ == "__main__":
    print("--- Starting GPU Status Generation ---")
    if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR, exist_ok=True)

    data = get_gpu_data()
    organized = organize_data_by_type(data)
    
    today = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(ARCHIVE_DIR, f"gpu_status_{today}.png")
    
    create_status_image(organized, out_file)
    create_gif(ARCHIVE_DIR, GIF_FILENAME)
    print("--- Finished ---")
