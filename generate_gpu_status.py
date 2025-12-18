import subprocess
import re
from collections import defaultdict

def get_slurm_data():
    """
    Runs 'scontrol show node' once to capture the state of all nodes.
    Returns a list of dictionaries containing node data.
    """
    try:
        # Run scontrol once for the whole cluster (much faster than looping)
        # -o (oneline) makes parsing slightly easier usually, but standard output is fine too.
        # We stick to standard output as per your grep example.
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
    # We ignore the first split if it's empty
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
        # We look for the number at the very end of the gpu string
        gres_match = re.search(r"Gres=.*gpu:.*?:?(\d+)", block)
        
        # If no Gres line, or no number found, assume 0
        if gres_match:
            gpu_count = int(gres_match.group(1))
        else:
            # Fallback: check for simple "gpu:8" format without subtypes
            simple_match = re.search(r"Gres=gpu:(\d+)", block)
            gpu_count = int(simple_match.group(1)) if simple_match else 0

        # Filter: We only care about GPU nodes for this report
        if gpu_count > 0:
            nodes.append({
                "name": node_name,
                "features": features,
                "gpu_count": gpu_count
            })
            
    return nodes

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
        mode = max(set(counts), key=counts.count)
        expected_counts[feature] = mode

    # --- REPORTING PHASE ---
    print(f"\n{'NODE':<12} {'STATUS':<10} {'GPU_COUNT':<12} {'EXPECTED':<10} {'HARDWARE_CLASS'}")
    print("-" * 80)
    
    # Sort by node name for clean output
    nodes.sort(key=lambda x: x['name'])
    
    for node in nodes:
        actual = node['gpu_count']
        expected = expected_counts[node['features']]
        
        # Determine Status
        if actual < expected:
            status = "\033[91mDEGRADED\033[0m" # Red Text
        elif actual > expected:
            status = "\033[93mOVER\033[0m"     # Yellow Text (Unusual but probably fine)
        else:
            status = "\033[92mOK\033[0m"       # Green Text
            
        # Truncate features for display if they are too long
        display_features = (node['features'][:30] + '..') if len(node['features']) > 30 else node['features']
        
        print(f"{node['name']:<12} {status:<19} {str(actual):<12} {str(expected):<10} {display_features}")

if __name__ == "__main__":
    generate_report()
