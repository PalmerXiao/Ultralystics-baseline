import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(data, title="Model Performance Comparison", 
                           xlabel="Number of Parameters (M)", ylabel="mAP50 (%)", 
                           size_scale_factor=20, output_filename="model_performance.png",
                           figsize=(12, 8), annotate_points=True):
    """
    Generates a scatter plot comparing model performance.

    Args:
        data (list): A list of dictionaries, where each dictionary contains:
                     'name': str (model name)
                     'map50': float (mAP50 score)
                     'gflops': float (GFLOPs)
                     'params': int (number of parameters)
                     'group': str (e.g., 'YOLOv5', 'YOLOv8', 'Custom') for coloring
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        size_scale_factor (float): Factor to scale GFLOPs for point size.
        output_filename (str): Name of the file to save the plot.
        figsize (tuple): Figure size (width, height) in inches.
        annotate_points (bool): Whether to annotate points with model names.
    """
    if not data:
        print("No data provided to plot.")
        return

    names = [d['name'] for d in data]
    map50_values = [d['map50'] for d in data]
    # Convert parameters to Millions
    params_m = [d['params'] / 1_000_000 for d in data]
    gflops_values = [d['gflops'] for d in data]
    groups = [d.get('group', 'Default') for d in data] # Use 'Default' if group is not specified

    # Create a color map for groups
    unique_groups = sorted(list(set(groups)))
    if not unique_groups: # Should not happen if data is present
        colors_cmap = plt.cm.get_cmap('tab10', 1)
    else:
        colors_cmap = plt.cm.get_cmap('tab10', len(unique_groups) if len(unique_groups) >= 1 else 1) # Ensure at least 1 color
    
    group_to_color = {group: colors_cmap(i) for i, group in enumerate(unique_groups)}
    point_colors = [group_to_color[g] for g in groups]

    # Scale GFLOPs for scatter plot size
    point_sizes = [g * size_scale_factor for g in gflops_values]

    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics
    plt.figure(figsize=figsize)

    scatter = plt.scatter(params_m, map50_values, s=point_sizes, c=point_colors, alpha=0.7, edgecolors='w', linewidth=0.5)

    if annotate_points:
        for i, name in enumerate(names):
            plt.annotate(name, (params_m[i], map50_values[i]),
                         textcoords="offset points", xytext=(5,5), ha='left', fontsize=12)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Create a legend for groups if there are multiple groups
    if len(unique_groups) > 1 and not (len(unique_groups) == 1 and unique_groups[0] == 'Default'):
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=group, markersize=10, 
                                     markerfacecolor=group_to_color[group]) 
                           for group in unique_groups]
        plt.legend(handles=legend_elements, title="Model Series", loc="lower right", fontsize=16, title_fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")
    
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Data for VisDrone dataset as provided by the user
    visdrone_data = [
        {
            'name': 'YOLOv5n', 'map50': 33.0, 'gflops': 7.1, 
            'params': 2504894, 'group': 'YOLOv5'
        },
        {
            'name': 'YOLOv5s', 'map50': 39.0, 'gflops': 23.8, 
            'params': 9115406, 'group': 'YOLOv5'
        },
        {
            'name': 'YOLOv5m', 'map50': 42.1, 'gflops': 64.0, 
            'params': 25051006, 'group': 'YOLOv5'
        },
        {
            'name': 'YOLOv6n', 'map50': 30.3, 'gflops': 11.8, 
            'params': 4234734, 'group': 'YOLOv6'
        },
        {
            'name': 'YOLOv6s', 'map50': 39.0, 'gflops': 23.8, 
            'params': 16299374, 'group': 'YOLOv6'
        },
        {
            'name': 'YOLOv6m', 'map50': 42.1, 'gflops': 64.0, 
            'params': 51981550, 'group': 'YOLOv6'
        },
        {
            'name': 'YOLOv8n', 'map50': 33.5, 'gflops': 8.1, 
            'params': 3007598, 'group': 'YOLOv8'
        },
        {
            'name': 'YOLOv8s', 'map50': 39.6, 'gflops': 28.5, 
            'params': 11129454, 'group': 'YOLOv8'
        },
        {
            'name': 'YOLOv8m', 'map50': 42.6, 'gflops': 78.7, 
            'params': 25845550, 'group': 'YOLOv8'
        },
        {
            'name': 'BiGA-YOLO-n', 'map50': 34.2, 'gflops': 7.6, 
            'params': 2140771, 'group': 'BiGA-YOLO'
        },
        {
            'name': 'BiGA-YOLO-s', 'map50': 41.0, 'gflops': 26.7, 
            'params': 7935779, 'group': 'BiGA-YOLO'
        },
        {
            'name': 'BiGA-YOLO-m', 'map50': 44.2, 'gflops': 74.8, 
            'params': 18811555, 'group': 'BiGA-YOLO'
        },
        # --- Add more models for VisDrone or other datasets here ---
        # Example:
        # {
        #     'name': 'RT-DETR-R34', 'map50': 47.0, 'gflops': 100.0, # Fictional GFLOPs for example size
        #     'params': 35000000, 'group': 'RT-DETR'
        # },
        # {
        #     'name': 'YOLOv10-N', 'map50': 35.0, 'gflops': 10.0,
        #     'params': 4000000, 'group': 'YOLOv10'
        # }
    ]

    plot_model_performance(
        visdrone_data,
        title="Performance on VisDrone Dataset (mAP50 vs Parameters)",
        output_filename="visdrone_performance_comparison.png",
        size_scale_factor=15 # Adjust this factor to get desired GFLOPs circle sizes
    )

    # --- Example for another dataset (fictional data) ---
    # coco_data = [
    #     {'name': 'ModelA-Small', 'map50': 30.0, 'gflops': 10.0, 'params': 5000000, 'group': 'ModelA'},
    #     {'name': 'ModelA-Large', 'map50': 45.0, 'gflops': 50.0, 'params': 20000000, 'group': 'ModelA'},
    #     {'name': 'ModelB-Fast', 'map50': 35.0, 'gflops': 5.0, 'params': 3000000, 'group': 'ModelB'},
    # ]
    # plot_model_performance(
    #     coco_data,
    #     title="Performance on COCO Dataset (mAP50 vs Parameters)",
    #     output_filename="coco_performance_comparison.png",
    #     size_scale_factor=25
    # ) 