import subprocess
import re
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import math

# --- HELPER FUNCTIONS ---

def extract_model_name(features_str):
    """
    Attempts to pick the most readable GPU model name from the feature string.
    e.g., "RTX_2080,rtx_2080,RTX_2080Ti,rtx_2080ti" -> "RTX 2080Ti"
    """
    features = features_str.split(',')
    candidates = []
    for f in features:
        # Filter out generic tags and lowercase duplicates
        if f.lower() in ['gpu', 'location=local']: continue
        if not f[0].isupper(): continue # Prefer capitalized ones
        if '=' in f: continue # Ignore k=v tags
        
        # Clean up display format (RTX_2080Ti -> RTX 2080Ti)
        clean_name = f.replace('_', ' ')
        candidates.append(clean_name)
        
    if not candidates:
        return features_str[:15] # Fallback to raw string truncated

    # Heuristic: The longest specific name is usually the best descriptor
    # e.g., between RTX 2080 and RTX 2080Ti, pick the latter.
    best_name = max(candidates, key=len)
    return best_name

# --- DATA GATHERING ---

def get_slurm_data():
    """Runs 'scontrol show node' once."""
    try:
        cmd = ["scontrol", "show", "node"]
        result = subprocess.check_output(cmd, encoding='utf-8', stderr=subprocess.DEVNULL)
        return result
    except Exception as e:
        print(f"Error running scontrol: {e}")
        return ""

def parse_nodes(raw_output):
    """Parses raw output into structured data."""
    nodes = []
    raw_blocks = raw_output.split("NodeName=")
    
    for block in raw_blocks:
        if not block.strip(): continue
        node_name = block.split()[0]
        
        # Extract raw features string for grouping logic
        feat_match = re.search(r"AvailableFeatures=([^\s]+)", block)
        raw_features = feat_match.group(1) if feat_match else "Unknown"
        
        # Extract pretty model name for display
        model_display_name = extract_model_name(raw_features)
        
        # Extract GPU Count
        gres_match = re.search(r"Gres=.*gpu:.*?:?(\d+)", block)
        if gres_match:
            gpu_count = int(gres_match.group(1))
        else:
            simple_match = re.search(r"Gres=gpu:(\d+)", block)
            gpu_count = int(simple_match.group(1)) if simple_match else 0

        if gpu_count > 0:
            nodes.append({
                "name": node_name,
                "raw_features": raw_features,
                "model_name": model_display_name,
                "gpu_count": gpu_count
            })
            
    return nodes

# --- VISUALIZATION (NEW SLOT LOGIC) ---

def save_cluster_image(nodes, expected_counts):
    if not os.path.exists("images"):
        os.makedirs("images")

    nodes.sort(key=lambda x: x['name'])

    # Grid Setup
    cols = 5
    rows = math.ceil(len(nodes) / cols)
    
    fig_height = max(6, rows * 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Canvas coordinate space
    ax.set_xlim(0, cols * 3.0)
    ax.set_ylim(0, rows * 2.5)
    ax.axis('off')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Cluster GPU Slots - {timestamp}", fontsize=16, fontweight='bold', y=0.99)

    # Node Container Dimensions
    node_w = 2.6
    node_h = 2.0
    spacing_x = 0.4
    spacing_y = 0.5
    start_x = 0.2
    start_y = (rows * 2.5) - 2.2

    # Colors
    c_node_bg = '#e0e0e0' # Light grey chassis
    c_node_border = '#777777'
    c_led_ok = '#2ecc71'   # Nice green
    c_led_miss = '#e74c3c' # Nice red
    c_text = '#2c3e50'

    for i, node in enumerate(nodes):
        curr_row = i // cols
        curr_col = i % cols
        
        # Node container position (bottom-left corner)
        nx = start_x + (curr_col * (node_w + spacing_x))
        ny = start_y - (curr_row * (node_h + spacing_y))

        # 1. Draw Node Chassis Container
        chassis = patches.FancyBboxPatch((nx, ny), node_w, node_h, 
                                       boxstyle="round,pad=0.1", 
                                       linewidth=1.5, edgecolor=c_node_border, facecolor=c_node_bg)
        ax.add_patch(chassis)

        # Node Name & Model Text
        ax.text(nx + node_w/2, ny + node_h - 0.3, node['name'], 
                ha='center', fontsize=11, fontweight='bold', color=c_text)
        ax.text(nx + node_w/2, ny + node_h - 0.6, node['model_name'], 
                ha='center', fontsize=9, fontstyle='italic', color=c_text)

        # 2. Draw GPU Slots (LEDs) inside the chassis
        actual = node['gpu_count']
        expected = expected_counts[node['raw_features']]
        
        # LED Layout definition
        led_rows = 2
        led_cols = math.ceil(expected / led_rows)
        led_w = 0.4
        led_h = 0.3
        led_pad_x = 0.1
        led_pad_y = 0.15
        
        # Start position for LEDs (bottom left inside chassis)
        led_start_x = nx + (node_w - (led_cols*(led_w+led_pad_x)))/2 + led_pad_x/2
        led_start_y = ny + 0.3

        slot_idx = 0
        for lr in range(led_rows -1, -1, -1): # Bottom row first
            for lc in range(led_cols):
                if slot_idx >= expected: break # Don't draw extra slots if expected is odd number

                lx = led_start_x + (lc * (led_w + led_pad_x))
                ly = led_start_y + (lr * (led_h + led_pad_y))
                
                # Color logic: Green if present, Red if missing slot
                if slot_idx < actual:
                    led_color = c_led_ok
                    edge_color = '#27ae60'
                else:
                    led_color = c_led_miss
                    edge_color = '#c0392b'

                # Draw LED rect
                led = patches.Rectangle((lx, ly), led_w, led_h, linewidth=1, 
                                      edgecolor=edge_color, facecolor=led_color)
                ax.add_patch(led)
                slot_idx += 1

    # Save File
    filename = f"images/status_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
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

    # Learning Phase (Peer Comparison on raw feature strings)
    feature_groups = defaultdict(list)
    for node in nodes:
        feature_groups[node['raw_features']].append(node['gpu_count'])
    
    expected_counts = {}
    for feature, counts in feature_groups.items():
        if not counts: continue
        mode = max(set(counts), key=counts.count)
        expected_counts[feature] = mode

    # Text Report
    print(f"\n{'NODE':<12} {'STATUS':<12} {'MODEL':<20} {'SLOTS (ACT/EXP)'}")
    print("-" * 65)
    nodes.sort(key=lambda x: x['name'])
    for node in nodes:
        actual = node['gpu_count']
        expected = expected_counts[node['raw_features']]
        
        if actual < expected:
            status = "\033[91mDEGRADED\033[0m"
        elif actual > expected:
            status = "\033[93mOVER\033[0m"
        else:
            status = "\033[92mOK\033[0m"
            
        slots = f"{actual}/{expected}"
        print(f"{node['name']:<12} {status:<21} {node['model_name']:<20} {slots:<15}")

    # Image Generation
    save_cluster_image(nodes, expected_counts)

if __name__ == "__main__":
    generate_report()
