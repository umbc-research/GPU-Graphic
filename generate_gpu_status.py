import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
import glob
from PIL import Image
import sys

# =================CONFIGURATION=================
# Comma-separated list of your GPU partition names
PARTITIONS = "gpu" 

# Directory to store daily images
ARCHIVE_DIR = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_archive"

# Filename for the combined historical GIF
GIF_FILENAME = "/umbc/rs/pi_doit/users/elliotg2/gpuGraphics/gpu_status_history.gif"

# Command timeout in seconds (prevent hanging on down nodes)
TIMEOUT = 45
# ===============================================


def run_command(command):
    """Executes a shell command and returns stdout and return code."""
    try:
        # print(f"Executing: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return "", 124 # Standard timeout exit code


def get_gpu_data():
    """
    Queries SLURM and compute nodes to gather GPU status data.
    Returns a dictionary structured by node.
    """
    data = {}
    print("Fetching node list from SLURM...")
    # 1. Get list of nodes, their state, and generic resources (GRES)
    # Format: NodeName|State|Gres
    sinfo_cmd = f"sinfo -p {PARTITIONS} -h -o '%N|%t|%G'"
    stdout, rc = run_command(sinfo_cmd)
    if rc != 0:
        print(f"Error running sinfo: {stdout}")
        sys.exit(1)

    print(f"Found {len(stdout.splitlines())} nodes. Querying GPUs on each node...")
    for line in stdout.split('\n'):
        if not line: continue
        parts = line.split('|')
        if len(parts) < 3: continue
        node_name, node_state, gres = parts[0], parts[1], parts[2]

        data[node_name] = {
            "slurm_state": node_state,
            "gpus": []
        }

        # 2. Query GPUs directly on the node using nvidia-smi via srun
        # We use --exclusive to ensure we get the node even if it's busy, 
        # but this might need adjustment based on your cluster's policy.
        nvidia_cmd = f"srun -w {node_name} -n1 -N1 --exclusive nvidia-smi --query-gpu=name,pci.bus_id,index --format=csv,noheader"
        gpu_stdout, gpu_rc = run_command(nvidia_cmd)

        if gpu_rc == 0 and gpu_stdout:
            # nvidia-smi ran successfully
            for gpu_line in gpu_stdout.split('\n'):
                if not gpu_line: continue
                gpu_parts = gpu_line.split(',')
                if len(gpu_parts) != 3: continue
                gpu_model, bus_id, index = [p.strip() for p in gpu_parts]

                # --- GPU State Logic ---
                # A GPU is "bad" if the node is down or drained in SLURM.
                # You can add more complex checks here (e.g., parsing nvidia-smi errors).
                state = "good"
                if "drain" in node_state.lower() or "down" in node_state.lower() or "*" in node_state:
                     state = "bad"
                
                data[node_name]["gpus"].append({
                    "model": gpu_model,
                    "bus_id": bus_id,
                    "index": index,
                    "state": state
                })
        else:
            # srun failed (e.g., node is completely down or unreachable).
            # Fallback: Attempt to parse GRES to define "ghost" bad GPUs.
            print(f"  Could not query {node_name} (State: {node_state}). Marking GPUs as bad.")
            gpu_count = 0
            gpu_model = "Unknown GPU"
            if "gpu" in gres:
                try:
                    # Example GRES formats: gpu:4, gpu:v100:4
                    gres_parts = gres.split(':')
                    for i, part in enumerate(gres_parts):
                        if part == 'gpu':
                            # Check for gpu:type:count
                            if i + 2 < len(gres_parts) and gres_parts[i+2].isdigit():
                                gpu_model = gres_parts[i+1].upper()
                                gpu_count = int(gres_parts[i+2])
                            # Check for gpu:count
                            elif i + 1 < len(gres_parts) and gres_parts[i+1].isdigit():
                                gpu_count = int(gres_parts[i+1])
                            break
                except Exception as e:
                    print(f"  Failed to parse GRES for {node_name}: {e}")

            for i in range(gpu_count):
                 data[node_name]["gpus"].append({
                    "model": gpu_model,
                    "bus_id": f"N/A",
                    "index": str(i),
                    "state": "bad"
                })

    return data


def organize_data_by_type(raw_data):
    """Reorganizes data from node-centric to GPU-type-centric for plotting."""
    organized = {}
    for node_name, node_info in raw_data.items():
        for gpu in node_info["gpus"]:
            model = gpu["model"]
            # Clean up model names
            model = model.replace("NVIDIA", "").replace("GeForce", "").strip()
            if model not in organized:
                organized[model] = {}
            if node_name not in organized[model]:
                organized[model][node_name] = []
            organized[model][node_name].append(gpu)
    # Sort by model name
    return dict(sorted(organized.items()))


def create_status_image(data, filename):
    """Generates a diagram of GPU status using matplotlib."""
    if not data:
        print("No data to plot.")
        return

    # Dynamic height based on number of GPU types
    fig_height = max(6, len(data) * 4)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_axis_off()
    
    # Title and Date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.98, "GPU Infrastructure Status Diagram", ha='center', va='top', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.95, f"Last Updated: {current_date}", ha='center', va='top', fontsize=14)

    # Legend
    good_patch = patches.Patch(color='green', label='Good State')
    bad_patch = patches.Patch(color='red', label='Bad State')
    ax.legend(handles=[good_patch, bad_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

    y_offset = 0.88
    SECTION_SPACING = 0.25
    SERVER_ROW_SPACING = 0.18
    
    for gpu_type, nodes in data.items():
        # --- Section Header ---
        ax.text(0.02, y_offset, gpu_type, fontsize=18, fontweight='bold', va='center')
        ax.add_patch(patches.Rectangle((0.02, y_offset - 0.01), 0.96, 0.002, color='black')) # Underline

        # --- Server Block Layout ---
        x_start = 0.25
        x_offset = x_start
        nodes_per_row = 5
        node_count = 0
        server_y = y_offset - 0.08
        
        sorted_nodes = sorted(nodes.items()) # Sort servers alphabetically

        for node_name, gpus in sorted_nodes:
            if node_count > 0 and node_count % nodes_per_row == 0:
                server_y -= SERVER_ROW_SPACING
                x_offset = x_start
            
            # Server Container Box
            SERVER_WIDTH = 0.13
            SERVER_HEIGHT = 0.10
            server_box = patches.Rectangle((x_offset, server_y - SERVER_HEIGHT), SERVER_WIDTH, SERVER_HEIGHT, 
                                           linewidth=1.5, edgecolor='#555555', facecolor='#f0f0f0')
            ax.add_patch(server_box)
            ax.text(x_offset + 0.005, server_y - 0.005, node_name, fontsize=10, fontweight='bold', va='top')
            
            # --- GPU Block Layout within Server ---
            gpu_x = x_offset + 0.005
            gpu_y = server_y - 0.035
            GPU_WIDTH = 0.028
            GPU_HEIGHT = 0.025
            GPUS_PER_ROW = 4
            
            for i, gpu in enumerate(gpus):
                color = '#28a745' if gpu['state'] == 'good' else '#dc3545' # Bootstrap green/red
                
                # GPU Status Block
                gpu_box = patches.Rectangle((gpu_x, gpu_y - GPU_HEIGHT), GPU_WIDTH, GPU_HEIGHT, 
                                            linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(gpu_box)
                
                # Label with GPU Index (e.g., "0", "1")
                label = gpu['index']
                ax.text(gpu_x + GPU_WIDTH/2, gpu_y - GPU_HEIGHT/2, label, 
                        fontsize=8, ha='center', va='center', color='white', fontweight='bold')
                
                gpu_x += GPU_WIDTH + 0.002
                if (i + 1) % GPUS_PER_ROW == 0:
                    gpu_x = x_offset + 0.005
                    gpu_y -= GPU_HEIGHT + 0.005

            x_offset += SERVER_WIDTH + 0.02
            node_count += 1

        # Calculate vertical space used by this section's servers
        rows_used = (node_count - 1) // nodes_per_row + 1
        y_offset = server_y - SERVER_ROW_SPACING # Move down for next section

    plt.tight_layout()
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Image generated successfully: {filename}")


def create_gif(image_folder, gif_filename):
    """Combines all PNGs in a folder into an animated GIF."""
    # Find all png files, sorted by name (which should be by date)
    filenames = sorted(glob.glob(os.path.join(image_folder, "gpu_status_*.png")))
    
    if not filenames:
        print("No images found in archive to create GIF.")
        return

    # Keep only the last 30 days to prevent the GIF from getting too huge
    filenames = filenames[-30:] 

    images = []
    for filename in filenames:
        try:
            images.append(Image.open(filename))
        except Exception as e:
            print(f"Failed to open image {filename}: {e}")

    if images:
        # Ensure directory exists
        os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
        
        # Save as GIF. duration is milliseconds per frame. loop=0 means infinite loop.
        images[0].save(
            gif_filename,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=800, # 0.8 seconds per frame
            loop=0
        )
        print(f"Historical GIF updated: {gif_filename}")
    else:
        print("No valid images to create GIF.")


# =================MAIN=================
if __name__ == "__main__":
    print("--- Starting GPU Status Generation ---")
    
    # 1. Ensure archive directory exists
    if not os.path.exists(ARCHIVE_DIR):
        print(f"Creating archive directory: {ARCHIVE_DIR}")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 2. Gather Data
    try:
        raw_data = get_gpu_data()
    except Exception as e:
        print(f"Critical error gathering data: {e}")
        sys.exit(1)

    # 3. Organize Data
    organized_data = organize_data_by_type(raw_data)
    
    # 4. Generate Daily Image
    today_str = datetime.now().strftime("%Y%m%d")
    daily_image_path = os.path.join(ARCHIVE_DIR, f"gpu_status_{today_str}.png")
    
    try:
        create_status_image(organized_data, daily_image_path)
    except Exception as e:
         print(f"Critical error generating image: {e}")
         sys.exit(1)
    
    # 5. Update Historical GIF
    try:
        create_gif(ARCHIVE_DIR, GIF_FILENAME)
    except Exception as e:
        print(f"Critical error creating GIF: {e}")

    print("--- Finished ---")
