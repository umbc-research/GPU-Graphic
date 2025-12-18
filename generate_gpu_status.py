import subprocess
import re
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# --- DATA GATHERING ---

def get_slurm_data():
    """
    Runs 'scontrol show node' once to capture the state of all nodes.
    Returns the raw string output.
    """
    try:
        # Run scontrol once for the whole cluster
        cmd = ["scontrol", "show", "node"]
        result = subprocess.check_output(cmd, encoding='utf-8')
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running scontrol: {e}")
        return ""

def parse_nodes(raw_output):
    """
    Parses the raw scontrol output into structured data.
    """
    nodes = []
    # Split output by "NodeName=" to isolate node blocks
    raw_blocks = raw_output.split("NodeName=")
    
    for block in raw_blocks:
        if not block.strip():
            continue
            
        node_name = block.split()[0]
        
        # Extract Features (Hardware Type)
        # Matches: AvailableFeatures=RTX_2080,rtx_2080...
        feat_match = re.search(r"AvailableFeatures=([^\s]+)", block)
        features = feat_match.group(1) if feat_match else "Unknown"
        
        # Extract GPU Count (Gres)
        # Matches: Gres=gpu:8  OR  Gres=gpu:rtx2080ti:8 (handles variants)
        gres_match = re.search(r"Gres=.*gpu:.*?:?(\d+)", block)
        
        # If no Gres line or complex match, check simple format
        if gres_match:
            gpu_count = int(gres_match.group(1))
        else:
            simple_match = re.search(r"Gres=gpu:(\d+)", block)
            gpu_count = int(simple_match.group(1)) if simple_match else 0

        # Filter: We only care about nodes that actually have GPUs
        if gpu_count > 0:
            nodes.append({
                "name": node_name,
                "features": features,
                "gpu_count": gpu_count
            })
            
    return nodes

# --- VISUALIZATION ---

def save_cluster_image(nodes, expected_counts):
    """
    Generates a visual map of the cluster status and saves it to ./images/
    """
    # Ensure images directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Sort nodes for consistent placement
    nodes.sort(key=lambda x: x['name'])

    # Grid Configuration
    total_nodes = len(nodes)
    cols = 5
    rows = (total_nodes // cols) + (1 if total_nodes % cols > 0 else 0)
    
    # Figure setup (dynamic height based on rows)
    fig_height = max(6, rows * 1.5) 
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Set plot limits to enclose the grid
    # Coordinates: x goes 0 -> cols*2, y goes 0 -> rows*2
    ax.set_xlim(0, cols * 2)
    ax.set_ylim(0, rows * 2)
    ax.axis('off') # Hide axes

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Cluster GPU Health - {timestamp}", fontsize=16, fontweight='bold', y=0.98)

    # Box dimensions
    box_width = 1.6
    box_height = 1.2
    spacing_x = 0.4
    spacing_y = 0.5
    
    start_x = 0.2
    start_y = (rows * 2) - 1.5 # Start from top

    for i, node in enumerate(nodes):
        # Calculate Grid Position
        curr_row = i // cols
        curr_col = i % cols
        
        x = start_x + (curr_col * (box_width + spacing_x))
        y = start_y - (curr_row * (box_height + spacing_y))

        # Determine Status & Color
        actual = node['gpu_count']
        expected = expected_counts[node['features']]
        
        if actual < expected:
            color = '#ff6b6b' # Red (Degraded)
            status_text = f"DEGRADED\n{actual}/{expected}"
        elif actual > expected:
            color = '#feca57' # Yellow (Over)
            status_text = f"OVER\n{actual}/{expected}"
        else:
            color = '#1dd1a1' # Green (OK)
            status_text = f"OK\n{actual}/{expected}"

        # Draw Node Box
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor='#333333', facecolor=color)
        ax.add_patch(rect)

        # Add Text Labels
        # Node Name
        ax.text(x + box_width/2, y + box_height*0.75, node['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
        # GPU Count/Status
        ax.text(x + box_width/2, y + box_height*0.35, status_text, 
                ha='center', va='center', fontsize=9, color='#333333')

    # Save File
    filename = f"images/status_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Snapshot saved to: {filename}")

# --- MAIN LOGIC ---

def generate_report():
    print("Gathering cluster status via scontrol...")
    raw_data = get_slurm_data()
    nodes = parse_nodes(raw_data)
    
    if not nodes:
        print("No GPU nodes found.")
        return

    # --- LEARNING PHASE (Peer Comparison) ---
    # Group nodes by their hardware features to find the "Mode" (expected count)
    feature_groups = defaultdict(list)
    for node in nodes:
        feature_groups[node['features']].append(node['gpu_count'])
    
    expected_counts = {}
    for feature, counts in feature_groups.items():
        # The most frequent GPU count in this group is assumed to be the correct one
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        expected_counts[feature] = mode

    # --- TEXT REPORTING PHASE ---
    print(f"\n{'NODE':<12} {'STATUS':<10} {'GPU_COUNT':<12} {'EXPECTED':<10} {'HARDWARE_CLASS'}")
    print("-" * 80)
    
    nodes.sort(key=lambda x: x['name'])
    
    for node in nodes:
        actual = node['gpu_count']
        expected = expected_counts[node['features']]
        
        # Determine Status
        if actual < expected:
            status = "\033[91mDEGRADED\033[0m" # Red Text
        elif actual > expected:
            status = "\033[93mOVER\033[0m"     # Yellow Text
        else:
            status = "\033[92mOK\033[0m"       # Green Text
            
        display_features = (node['features'][:30] + '..') if len(node['features']) > 30 else node['features']
        print(f"{node['name']:<12} {status:<19} {str(actual):<12} {str(expected):<10} {display_features}")

    # --- IMAGE GENERATION PHASE ---
    save_cluster_image(nodes, expected_counts)

if __name__ == "__main__":
    generate_report()
