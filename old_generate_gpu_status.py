import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def generate_gpu_diagram(gpu_states, output_file='gpu_status.png'):
    """
    gpu_states: List of dicts with 'server_name', 'bus_id', 'state', and 'type'
    """
    
    server_configs = {
        'RTX 2080Ti': {'count': 4, 'gpus': 8},
        'RTX 6000': {'count': 7, 'gpus': 8},
        'RTX 8000': {'count': 2, 'gpus': 8},
        'L40S': {'count': 10, 'gpus': 4},
        'H100': {'count': 2, 'gpus': 2}
    }
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    
    current_y = 0
    padding = 0.5
    server_width = 10
    gpu_box_size = 0.8
    
    # Sort or iterate through types
    for gpu_type, config in server_configs.items():
        # Label for the section
        ax.text(-1, current_y - 0.5, gpu_type, fontsize=14, fontweight='bold', va='top', ha='right')
        
        num_servers = config['count']
        gpus_per_server = config['gpus']
        
        # Calculate how many servers per row for this section
        servers_per_row = 4 if num_servers > 4 else num_servers
        
        for i in range(num_servers):
            row = i // servers_per_row
            col = i % servers_per_row
            
            x_start = col * (server_width + 1)
            y_start = current_y - (row * 3.5)
            
            # Draw Server Box
            server_box_height = 2.5
            rect = patches.Rectangle((x_start, y_start - server_box_height), server_width, server_box_height, 
                                     linewidth=1, edgecolor='black', facecolor='whitesmoke')
            ax.add_patch(rect)
            
            server_name = f"{gpu_type.replace(' ', '_')}_Srv_{i+1}"
            ax.text(x_start + 0.2, y_start - 0.4, server_name, fontsize=8, fontweight='bold')
            
            # Draw GPUs inside server
            for g_idx in range(gpus_per_server):
                # Arrange GPUs in 2 rows if many
                g_row = g_idx // 4
                g_col = g_idx % 4
                
                gx = x_start + 0.5 + (g_col * 2.2)
                gy = y_start - 1.2 - (g_row * 1.0)
                
                # Mock GPU identifier
                bus_id = f"0000:0{g_idx}:00.0"
                
                # Check state
                state = gpu_states.get((server_name, bus_id), 'good')
                color = 'green' if state == 'good' else 'red'
                
                # Draw GPU box
                gpu_rect = patches.Rectangle((gx, gy - 0.6), 1.8, 0.6, color=color, alpha=0.8)
                ax.add_patch(gpu_rect)
                ax.text(gx + 0.9, gy - 0.3, f"GPU {g_idx}", color='white', ha='center', va='center', fontsize=6)

        # Update current_y for next section
        rows_in_section = (num_servers - 1) // servers_per_row + 1
        current_y -= (rows_in_section * 4) + 1

    ax.set_xlim(-5, 50)
    ax.set_ylim(current_y, 2)
    ax.axis('off')
    plt.title("GPU Infrastructure Status Diagram", fontsize=18, pad=20)
    
    # Legend
    green_patch = patches.Patch(color='green', label='Good State')
    red_patch = patches.Patch(color='red', label='Bad State')
    plt.legend(handles=[green_patch, red_patch], loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

# Create Mock Data
mock_states = {}
server_configs = {
    'RTX 2080Ti': {'count': 4, 'gpus': 8},
    'RTX 6000': {'count': 7, 'gpus': 8},
    'RTX 8000': {'count': 2, 'gpus': 8},
    'L40S': {'count': 10, 'gpus': 4},
    'H100': {'count': 2, 'gpus': 2}
}

for gpu_type, config in server_configs.items():
    for i in range(config['count']):
        server_name = f"{gpu_type.replace(' ', '_')}_Srv_{i+1}"
        for g_idx in range(config['gpus']):
            bus_id = f"0000:0{g_idx}:00.0"
            # Randomly assign a bad state to 5% of GPUs for demonstration
            mock_states[(server_name, bus_id)] = 'good' if random.random() > 0.05 else 'bad'

generate_gpu_diagram(mock_states)

