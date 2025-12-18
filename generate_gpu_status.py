import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
import datetime
import glob
import os
import imageio.v2 as imageio
import sys

# --- Configuration: Expected Specs ---
# Based on 
EXPECTED_SPECS = {
    "RTX 2080Ti": {
        "nodes": [f"g20-{i:02d}" for i in range(1, 5)], # g20-01 to g20-04
        "count": 8
    },
    "RTX 6000": {
        "nodes": [f"g20-{i:02d}" for i in range(5, 12)], # g20-05 to g20-11
        "count": 8
    },
    "RTX 8000": {
        "nodes": ["g20-12", "g20-13"],
        "count": 8
    },
    "L40S": {
        # g24-01 to g24-08 AND g24-11, g24-12
        "nodes": [f"g24-{i:02d}" for i in range(1, 9)] + ["g24-11", "g24-12"], 
        "count": 4
    },
    "H100": {
        "nodes": ["g24-09", "g24-10"],
        "count": 2
    },
}

def get_real_gpu_status():
    """
    Iterates through the expected topology and runs 'srun' to check
    the physical presence of GPUs on each node via nvidia-smi.
    """
    status_map = {}
    print(f"Starting Cluster GPU Scan at {datetime.datetime.now().strftime('%H:%M:%S')}...")

    for gpu_type, spec in EXPECTED_SPECS.items():
        print(f"Scanning {gpu_type} nodes...")
        for node in spec['nodes']:
            expected_count = spec['count']
            
            # Default to all Bad (False) until proven Good
            node_status = [False] * expected_count
            
            # Command to check GPU indices on the remote node
            # -w: target node
            # -N 1 -n 1: run one task
            # --quiet: suppress srun banners
            # The breakdown of the required flags
            cmd = [
                "srun",
                "--cluster=chip-gpu",      # Target the GPU cluster
                "--account=pi_doit",       # Use the privileged account
                "--partition=support",     # Use the hidden 'support' partition
                "--time=00:01:00",         # Set a short time limit (1 min is enough for nvidia-smi)
                "--mem=1G",                # Request minimal memory (1GB)
                "--gres=gpu:1",            # MANDATORY: You must request a GPU to touch the GPU node
                "--nodelist={node_name}", # (Presumably you are iterating over nodes here)
                "nvidia-smi"               # The actual command to run
            ]
            try:
                # Run with a timeout to prevent hanging on down/busy nodes
                # Adjust timeout (seconds) as needed based on cluster load
                output = subprocess.check_output(
                    cmd, shell=True, universal_newlines=True, timeout=10
                )
                
                # Parse output: list of indices found (e.g., "0\n1\n2...")
                found_indices = [int(x) for x in output.strip().split('\n') if x.strip().isdigit()]
                
                # Update status for found GPUs
                for idx in found_indices:
                    if idx < expected_count:
                        node_status[idx] = True
                    else:
                        print(f"  [WARN] Node {node} reported GPU index {idx} which exceeds expected {expected_count}")

            except subprocess.TimeoutExpired:
                print(f"  [ERR] Node {node}: Timed out (Node likely busy or unresponsive)")
                # Leaves node_status as all False (Red)
            
            except subprocess.CalledProcessError:
                print(f"  [ERR] Node {node}: Connection failed or nvidia-smi error (Node likely DOWN)")
                # Leaves node_status as all False (Red)

            status_map[node] = node_status
            
    return status_map

def draw_cluster_status(status_map, date_str):
    """
    Generates the visual diagram using Matplotlib.
    """
    fig_width = 24
    # Calculate height based on number of categories
    fig_height = len(EXPECTED_SPECS) * 3.5 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Styling
    plt.title(f"GPU Infrastructure Status - {date_str}", fontsize=28, pad=20)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, len(EXPECTED_SPECS) * 5)
    ax.axis('off')

    # Legend
    legend_elements = [
        patches.Patch(facecolor='green', edgecolor='black', label='Good State (Up)'),
        patches.Patch(facecolor='red', edgecolor='black', label='Bad State (Down/Missing)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

    y_offset = len(EXPECTED_SPECS) * 5 - 2
    
    for gpu_type, spec in EXPECTED_SPECS.items():
        # Draw Section Header
        ax.text(0.5, y_offset + 1.8, gpu_type, fontsize=20, fontweight='bold', va='center')
        
        # Draw Horizontal Divider
        ax.hlines(y_offset + 1.2, 0, 24, colors='gray', linestyles='solid', linewidth=0.5)

        x_offset = 2.5
        current_row_y = y_offset
        
        for node in spec['nodes']:
            # Get status (True=Good, False=Bad)
            gpu_states = status_map.get(node, [False] * spec['count'])
            
            # Container Box for Node
            box_width = max(2.2, spec['count'] * 0.5) 
            rect = patches.Rectangle((x_offset, current_row_y - 1.5), box_width, 2.2, 
                                     linewidth=1, edgecolor='black', facecolor='#f9f9f9')
            ax.add_patch(rect)
            
            # Node Label
            ax.text(x_offset + 0.1, current_row_y + 0.3, node, fontsize=10, fontweight='bold')
            
            # Draw individual GPUs
            gpu_x = x_offset + 0.2
            for i, is_good in enumerate(gpu_states):
                color = 'green' if is_good else 'red'
                gpu_rect = patches.Rectangle((gpu_x, current_row_y - 1.2), 0.35, 1.0, 
                                             linewidth=0, facecolor=color)
                ax.add_patch(gpu_rect)
                
                # GPU ID Text
                ax.text(gpu_x + 0.175, current_row_y - 1.4, f"G{i}", 
                        fontsize=7, color='black', ha='center', va='center')
                
                gpu_x += 0.45

            # Move to next node position
            x_offset += box_width + 0.5
            
            # Wrap to next line if too many nodes in one row
            if x_offset > 20:
                x_offset = 2.5
                current_row_y -= 2.8

        # Move Y offset down for the next Category
        y_offset -= 5

    # Save daily image
    filename = f"gpu_status_{date_str}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Generated status image: {filename}")
    return filename

def create_gif():
    """
    Compiles all gpu_status_*.png files into an animated GIF.
    """
    images = []
    # Sort primarily by date pattern inside filename
    filenames = sorted(glob.glob("gpu_status_*.png"))
    
    if not filenames:
        print("No images found to generate GIF.")
        return

    print(f"Compiling GIF from {len(filenames)} images...")
    for filename in filenames:
        images.append(imageio.imread(filename))
        
    # Save GIF (duration is seconds per frame)
    gif_name = 'gpu_health_trends.gif'
    imageio.mimsave(gif_name, images, duration=1.0, loop=0)
    print(f"GIF generated: {gif_name}")

# --- Execution ---
if __name__ == "__main__":
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 1. Gather Data (LIVE)
    current_status = get_real_gpu_status()
    
    # 2. Generate Daily Image
    draw_cluster_status(current_status, today)
    
    # 3. Update Trend GIF
    create_gif()
