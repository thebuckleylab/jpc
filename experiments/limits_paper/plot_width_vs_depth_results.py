import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
})

# Plot styling constants
FIG_SIZE = (10, 8)
FONT_SIZES = {"label": 35, "legend": 25, "tick": 30}
LABEL_PAD = 20
ALPHA = 0.7


def _load_npy_safe(path, flatten=True):
    """Safely load numpy array, handling object arrays."""
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True)
    if flatten and isinstance(arr, np.ndarray):
        if arr.dtype == object:
            return np.array([x.item() if hasattr(x, 'item') else (float(x) if flatten else x) for x in arr])
        return arr.flatten()
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.array([g if isinstance(g, np.ndarray) else np.array(g) for g in arr])
    return arr


def discover_available_configs(results_dir, seed=0, param_type=None, use_skips=None):
    """Discover all available (width, n_hidden) configurations from the results directory.
    
    Args:
        results_dir: Directory to search
        seed: Seed value to match
        param_type: Parameter type to filter by ('sp' or 'mupc', None to load all)
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    
    Returns:
        Dictionary mapping (width, n_hidden) tuples to directory paths
    """
    configs = {}
    seed_str = str(seed)
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    for root, _, files in os.walk(results_dir):
        if "loss_rescalings.npy" in files:
            if root.split(os.sep)[-1] == seed_str:
                # Filter by param_type if specified
                if param_type_str is None or param_type_str in root:
                    # Filter by use_skips if specified
                    if use_skips_str is None or use_skips_str in root:
                        # Extract width and n_hidden from directory path
                        width = None
                        n_hidden = None
                        
                        for part in root.split(os.sep):
                            if part.endswith("_width"):
                                try:
                                    width = int(part.replace("_width", ""))
                                except ValueError:
                                    pass
                            elif part.endswith("_n_hidden"):
                                try:
                                    n_hidden = int(part.replace("_n_hidden", ""))
                                except ValueError:
                                    pass
                        
                        if width is not None:
                            configs[(width, n_hidden)] = root
    
    return configs


def extract_gamma_0_from_dir(results_dir, seed=0, param_type=None, use_skips=None):
    """Extract gamma_0 value from results directory.
    
    Looks for directories or files matching gamma_0 patterns.
    """
    seed_str = str(seed)
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    # Search for gamma_0 in directory paths
    for root, _, files in os.walk(results_dir):
        if root.split(os.sep)[-1] == seed_str:
            if param_type_str is None or param_type_str in root:
                if use_skips_str is None or use_skips_str in root:
                    for part in root.split(os.sep):
                        if part.endswith("_gamma_0"):
                            try:
                                gamma_0 = float(part.replace("_gamma_0", ""))
                                return gamma_0
                            except ValueError:
                                pass
    
    # Search for DMFT loss files with gamma_0 pattern
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.startswith("dmft_loss_") and file.endswith(f"_seed_{seed_str}.npy"):
                file_base = file.replace("dmft_loss_", "").replace(f"_seed_{seed_str}.npy", "")
                if file_base.endswith("_gamma_0"):
                    try:
                        gamma_0 = float(file_base.replace("_gamma_0", ""))
                        return gamma_0
                    except ValueError:
                        pass
    
    return None


def _plot_rescaling_vs_width_depth_single_seed(results_dir, plot_dir, seed, gamma_0, 
                                                param_type, output_dim, widths_filter, n_hiddens_filter, use_skips=None, linear_z_scale=False):
    """Helper function to plot a single 3D plot for a specific seed.
    
    Args:
        linear_z_scale: If True, use linear scale for z-axis. If False, use log10 scale (default).
    """
    # Discover all available configurations
    configs = discover_available_configs(results_dir, seed=seed, param_type=param_type, use_skips=use_skips)
    
    if not configs:
        print(f"  Warning: No configurations found for seed {seed}")
        return
    
    print(f"  Found {len(configs)} configurations for seed {seed}")
    
    # Show available L values before filtering
    available_L_values = set()
    for (w, n_hidden), path in configs.items():
        L = (n_hidden + 1) if n_hidden is not None else 1
        available_L_values.add(L)
    print(f"  Available depths (L): {sorted(available_L_values)}")
    
    # Filter configurations if filters are provided
    if widths_filter is not None and len(widths_filter) > 0:
        widths_filter_set = set(widths_filter)
        # Show what widths are available before filtering
        available_widths = {w for (w, h), path in configs.items()}
        print(f"  Available widths before filtering: {sorted(available_widths)}")
        print(f"  Filtering to widths: {sorted(widths_filter_set)}")
        configs = {(w, h): path for (w, h), path in configs.items() if w in widths_filter_set}
        print(f"  Filtered to {len(configs)} configurations matching widths")
        # Show what widths remain after filtering
        remaining_widths = {w for (w, h), path in configs.items()}
        print(f"  Remaining widths after filtering: {sorted(remaining_widths)}")
    
    if n_hiddens_filter is not None and len(n_hiddens_filter) > 0:
        # n_hiddens_filter is for n_hidden (number of hidden layers)
        n_hiddens_filter_set = set(n_hiddens_filter)
        # Show what n_hidden values are available before filtering
        available_n_hiddens_before = {n_hidden if n_hidden is not None else 0 for (w, n_hidden), path in configs.items()}
        print(f"  Available n_hidden values before filtering: {sorted(available_n_hiddens_before)}")
        print(f"  Filtering to n_hidden values: {sorted(n_hiddens_filter_set)}")
        filtered_configs = {}
        for (w, n_hidden), path in configs.items():
            n_hidden_val = n_hidden if n_hidden is not None else 0
            if n_hidden_val in n_hiddens_filter_set:
                filtered_configs[(w, n_hidden)] = path
        configs = filtered_configs
        print(f"  Filtered to {len(configs)} configurations matching n_hidden")
        # Show what n_hidden values remain after filtering
        remaining_n_hiddens = {n_hidden if n_hidden is not None else 0 for (w, n_hidden), path in configs.items()}
        print(f"  Remaining n_hidden values after filtering: {sorted(remaining_n_hiddens)}")
    
    # Load rescaling data for each configuration
    data_points = []
    widths = []
    depths = []
    rescalings_minus_one = []
    
    for (width, n_hidden), dir_path in configs.items():
        rescaling_path = os.path.join(dir_path, "loss_rescalings.npy")
        rescalings = _load_npy_safe(rescaling_path)  # Use default flatten=True
        
        if rescalings is not None and len(rescalings) > 0:
            # rescalings should be a 1D array after flattening
            first_rescaling = rescalings[0]
            
            # Convert to float, handling numpy scalars
            if isinstance(first_rescaling, np.ndarray):
                if first_rescaling.ndim == 0:
                    first_rescaling = float(first_rescaling.item())
                elif output_dim > 1:
                    # If it's a vector, compute norm
                    first_rescaling = float(np.linalg.norm(first_rescaling, ord=2))
                else:
                    first_rescaling = float(first_rescaling.item() if first_rescaling.size == 1 else first_rescaling[0])
            else:
                first_rescaling = float(first_rescaling)
            
            widths.append(width)
            # L = n_hidden + 1 (number of layers)
            L = (n_hidden + 1) if n_hidden is not None else 1
            depths.append(L)
            rescalings_minus_one.append(first_rescaling - 1.0)
            data_points.append((width, L, first_rescaling - 1.0))
    
    if not data_points:
        print("  Warning: No rescaling data available")
        return
    
    print(f"  Loaded {len(data_points)} data points")
    if len(data_points) > 0:
        print(f"  Width range: {min(widths)} - {max(widths)}")
        print(f"  L (layers) range: {min(depths)} - {max(depths)}")
        print(f"  Rescaling-1 range: {min(rescalings_minus_one):.6f} - {max(rescalings_minus_one):.6f}")
    
    # Create 3D plot
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to arrays
    widths_arr = np.array(widths)
    depths_arr = np.array(depths)
    rescalings_arr_original = np.array(rescalings_minus_one)  # Keep original for linear scale
    rescalings_arr = rescalings_arr_original.copy()  # Working copy
    
    # Use log base 2 for both width and L (matplotlib 3D doesn't support log scale directly)
    log2_widths = np.log2(widths_arr)
    log2_depths = np.log2(depths_arr)
    
    # Choose z-axis scale based on linear_z_scale parameter
    if linear_z_scale:
        # Use linear scale for z-axis (rescaling - 1)
        z_values = rescalings_arr_original
        min_rescaling = min(rescalings_arr_original)
        offset = 0.0
    else:
        # Use log base 10 for z-axis (rescaling - 1)
        # Ensure all values are positive for log scale
        min_rescaling = min(rescalings_arr)
        offset = 0.0
        if min_rescaling <= 0:
            print(f"  Warning: Some rescaling-1 values are non-positive (min={min_rescaling:.6f}), adding small offset for log scale")
            offset = abs(min_rescaling) + 1e-10
            rescalings_arr = rescalings_arr + offset
        z_values = np.log10(rescalings_arr)
    
    # Plot data points in blue - make them larger and ensure they're visible
    ax.scatter(log2_widths, log2_depths, z_values, 
               color='#1E88E5', s=200, alpha=0.9, label='Data', depthshade=True, edgecolors='darkblue', linewidths=1)
    
    # Plot theory plane: L / (gamma_0^2 * width)
    # Skip for 'sp' param_type
    if param_type != 'sp':
        # Calculate theory values at data point locations for projections
        widths_linear = 2**log2_widths
        depths_linear = 2**log2_depths
        # Theory: (L-1)/(gamma_0^2*N) without skips, 1/(gamma_0^2*N) + (L-1)/(gamma_0^2*N) with skips
        if use_skips:
            theory_values_linear = 1.0 / (gamma_0**2 * widths_linear) + (depths_linear - 1) / (gamma_0**2 * widths_linear)
        else:
            theory_values_linear = (depths_linear - 1) / (gamma_0**2 * widths_linear)
        if not linear_z_scale and min_rescaling <= 0:
            theory_values_linear = theory_values_linear + offset
        
        if linear_z_scale:
            theory_values_z = theory_values_linear
        else:
            theory_values_z = np.log10(theory_values_linear)
        
        # Draw dotted projection lines from data points to theory plane
        for i in range(len(log2_widths)):
            ax.plot([log2_widths[i], log2_widths[i]], 
                   [log2_depths[i], log2_depths[i]], 
                   [z_values[i], theory_values_z[i]],
                   'k--', alpha=0.8, linewidth=1, zorder=0)
    
    if param_type != 'sp':
        # Create mesh for the plane using log base 2 for both width and L
        min_width, max_width = min(widths_arr), max(widths_arr)
        min_depth, max_depth = min(depths_arr), max(depths_arr)
        
        log2_width_mesh = np.linspace(np.log2(min_width), np.log2(max_width), 50)
        log2_L_mesh = np.linspace(np.log2(min_depth), np.log2(max_depth), 50)
        W_log2, L_log2_grid = np.meshgrid(log2_width_mesh, log2_L_mesh)
        
        # Convert back to linear scale for theory calculation
        W_linear = 2**W_log2
        L_linear = 2**L_log2_grid
        # Theory: (L-1)/(gamma_0^2*N) without skips, 1/(gamma_0^2*N) + (L-1)/(gamma_0^2*N) with skips
        # Note: depth is now L (number of layers)
        if use_skips:
            Z_theory_linear = 1.0 / (gamma_0**2 * W_linear) + (L_linear - 1) / (gamma_0**2 * W_linear)
        else:
            Z_theory_linear = (L_linear - 1) / (gamma_0**2 * W_linear)
        
        # Convert theory to appropriate scale for z-axis
        if linear_z_scale:
            Z_theory = Z_theory_linear
        else:
            # Apply same offset if needed for log scale
            if min_rescaling <= 0:
                Z_theory_linear = Z_theory_linear + offset
            Z_theory = np.log10(Z_theory_linear)
        
        # Plot the surface
        # Try different parameter names for different matplotlib versions
        try:
            # Try with 'shade' parameter (older matplotlib versions)
            surf = ax.plot_surface(W_log2, L_log2_grid, Z_theory, alpha=0.3, color='gray', 
                                   shade=True, zorder=1)
        except (TypeError, AttributeError):
            # Fallback for newer matplotlib versions that use 'shading'
            try:
                surf = ax.plot_surface(W_log2, L_log2_grid, Z_theory, alpha=0.3, color='gray', 
                                       shading='auto', zorder=1)
            except (TypeError, AttributeError):
                # If both fail, try without shading parameter
                surf = ax.plot_surface(W_log2, L_log2_grid, Z_theory, alpha=0.3, color='gray', 
                                       zorder=1)
    
    # Set labels
    ax.set_xlabel("$N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_ylabel("$L$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if output_dim > 1:
        ax.set_zlabel(r"$\|\mathbf{S}(\boldsymbol{\theta}_t)\|_2 - 1$", 
                      fontsize=FONT_SIZES["label"], labelpad=34 )
    else:
        ax.set_zlabel(r"$s(\boldsymbol{\theta}_t) - 1$", 
                      fontsize=FONT_SIZES["label"], labelpad=34)
    
    # Set custom tick labels for x-axis (width) - show as many as there are depths
    num_depths = len(set(depths_arr))
    min_width_log2 = np.log2(min(widths_arr))
    max_width_log2 = np.log2(max(widths_arr))
    # Create evenly spaced tick positions matching the number of depths
    width_tick_positions = np.linspace(min_width_log2, max_width_log2, num_depths)
    # Round to nearest integers for cleaner labels
    width_tick_positions = np.round(width_tick_positions).astype(int)
    width_tick_labels = [f"$2^{{{p}}}$" for p in width_tick_positions]
    ax.set_xticks(width_tick_positions)
    ax.set_xticklabels(width_tick_labels)
    
    # Set custom tick labels for y-axis (L) - show powers of 2 in exponential notation
    min_depth_log2 = np.log2(min(depths_arr))
    max_depth_log2 = np.log2(max(depths_arr))
    # Find all powers of 2 in the range
    min_power = int(np.ceil(min_depth_log2))
    max_power = int(np.floor(max_depth_log2))
    depth_tick_positions = np.arange(min_power, max_power + 1)
    depth_tick_labels = [f"$2^{{{p}}}$" for p in depth_tick_positions]
    ax.set_yticks(depth_tick_positions)
    ax.set_yticklabels(depth_tick_labels)
    
    # Set custom tick labels for z-axis (rescaling - 1)
    if linear_z_scale:
        # Linear scale: show evenly spaced ticks
        min_rescaling_val = min(rescalings_arr_original)
        max_rescaling_val = max(rescalings_arr_original)
        z_tick_positions = np.linspace(min_rescaling_val, max_rescaling_val, 5)
        # Format with appropriate precision
        z_tick_labels = []
        for p in z_tick_positions:
            if abs(p) < 0.01 or abs(p) > 1000:
                z_tick_labels.append(f"${p:.2e}$")
            else:
                z_tick_labels.append(f"${p:.4f}$")
        ax.set_zticks(z_tick_positions)
        ax.set_zticklabels(z_tick_labels)
    else:
        # Log scale: show only 3 powers of 10 in exponential notation
        min_rescaling_log10 = min(z_values)
        max_rescaling_log10 = max(z_values)
        # Find 3 evenly spaced tick positions
        z_tick_positions = np.linspace(min_rescaling_log10, max_rescaling_log10, 3)
        # Round to nearest integers for cleaner labels
        z_tick_positions = np.round(z_tick_positions).astype(int)
        z_tick_labels = [f"$10^{{{p}}}$" for p in z_tick_positions]
        ax.set_zticks(z_tick_positions)
        ax.set_zticklabels(z_tick_labels)
    
    # Set tick parameters
    ax.tick_params(axis='x', labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis='y', labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis='z', labelsize=FONT_SIZES["tick"], pad=10)
    
    # Flip x-axis only
    ax.invert_xaxis()
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='#1E88E5', markersize=10, 
                          label='Data', alpha=ALPHA)]
    if param_type != 'sp':
        handles.append(Patch(facecolor='gray', alpha=0.3, 
                            label=r'Theory ($L/N$)'))
    ax.legend(handles=handles, fontsize=FONT_SIZES["legend"], loc='upper right')
    
    # Save plot
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"rescaling_vs_width_depth_seed_{seed}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', pad_inches=0.9)
    plt.close()
    
    print(f"  Saved plot to {os.path.join(plot_dir, filename)}")


def plot_rescaling_vs_width_depth(results_dir, plot_dir, seeds=[0, 1, 2], gamma_0=None, 
                                   param_type=None, output_dim=1, widths_filter=None, n_hiddens_filter=None, use_skips=None, linear_z_scale=False):
    """Plot PC rescaling - 1 as a function of width N and depth L (number of layers) in 3D.
    
    Creates a separate 3D plot for each seed in the provided list with:
    - Width (N) on x-axis
    - Depth L (number of layers = n_hidden + 1) on y-axis
    - Rescaling - 1 on z-axis
    - Data points in blue
    - Theory as a plane: L / (gamma_0^2 * width)
    
    Args:
        seeds: List of seed values to plot. One plot will be generated for each seed.
        n_hiddens_filter: Filter by n_hidden (number of hidden layers). L = n_hidden + 1 is computed during data loading.
        use_skips: Whether to filter by use_skips (True/False, None to load all)
        linear_z_scale: If True, use linear scale for z-axis. If False, use log10 scale (default).
    """
    if not seeds:
        print("  Warning: No seeds provided")
        return
    
    seeds_to_plot = sorted(seeds)
    print(f"  Creating plots for seeds: {seeds_to_plot} ({len(seeds_to_plot)} plots)")
    
    # Get gamma_0 from directory if not provided (use first seed)
    if gamma_0 is None:
        gamma_0 = extract_gamma_0_from_dir(results_dir, seed=seeds_to_plot[0], param_type=param_type, use_skips=use_skips)
        if gamma_0 is None:
            print("  Warning: gamma_0 not found, using default 2.0")
            gamma_0 = 2.0
        else:
            print(f"  Found gamma_0 = {gamma_0} from directory")
    else:
        print(f"  Using provided gamma_0 = {gamma_0}")
    
    # Create a plot for each seed
    for seed_val in seeds_to_plot:
        print(f"\n  Generating plot for seed {seed_val}...")
        _plot_rescaling_vs_width_depth_single_seed(
            results_dir, plot_dir, seed_val, gamma_0, 
            param_type, output_dim, widths_filter, n_hiddens_filter, use_skips, linear_z_scale
        )


def plot_error_heatmap(results_dir, plot_dir, seed=0, gamma_0=None, 
                       param_type=None, output_dim=1, widths_filter=None, n_hiddens_filter=None, use_skips=None):
    """Plot a heatmap showing absolute errors from the theoretical plane as a function of N and L.
    
    Creates a 2D heatmap with:
    - Width (N) on x-axis
    - Depth L (number of layers) on y-axis
    - Color representing absolute error: |actual_rescaling - theoretical_rescaling|
    
    Averages over multiple seeds (0, 1, 2) if available.
    """
    # Find all available seeds
    available_seeds = set()
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    for root, _, files in os.walk(results_dir):
        if "loss_rescalings.npy" in files:
            last_dir = root.split(os.sep)[-1]
            try:
                seed_val = int(last_dir)
                # Filter by param_type and use_skips if specified
                if param_type_str is None or param_type_str in root:
                    if use_skips_str is None or use_skips_str in root:
                        available_seeds.add(seed_val)
            except ValueError:
                pass
    
    if not available_seeds:
        print("  Warning: No seeds found")
        return
    
    seeds_to_use = sorted(available_seeds)
    print(f"  Found seeds: {seeds_to_use}, averaging over {len(seeds_to_use)} seeds")
    
    # Get gamma_0 from directory if not provided (use first available seed)
    if gamma_0 is None:
        gamma_0 = extract_gamma_0_from_dir(results_dir, seed=seeds_to_use[0], param_type=param_type, use_skips=use_skips)
        if gamma_0 is None:
            print("  Warning: gamma_0 not found, using default 2.0")
            gamma_0 = 2.0
        else:
            print(f"  Found gamma_0 = {gamma_0} from directory")
    else:
        print(f"  Using provided gamma_0 = {gamma_0}")
    
    # Discover all available configurations (using first seed to get structure)
    print("  Discovering available (width, n_hidden) configurations...")
    base_configs = discover_available_configs(results_dir, seed=seeds_to_use[0], param_type=param_type, use_skips=use_skips)
    
    if not base_configs:
        print("  Warning: No configurations found")
        return
    
    # Filter configurations if filters are provided
    if widths_filter is not None and len(widths_filter) > 0:
        widths_filter_set = set(widths_filter)
        base_configs = {(w, h): path for (w, h), path in base_configs.items() if w in widths_filter_set}
    
    if n_hiddens_filter is not None and len(n_hiddens_filter) > 0:
        n_hiddens_filter_set = set(n_hiddens_filter)
        filtered_configs = {}
        for (w, n_hidden), path in base_configs.items():
            n_hidden_val = n_hidden if n_hidden is not None else 0
            if n_hidden_val in n_hiddens_filter_set:
                filtered_configs[(w, n_hidden)] = path
        base_configs = filtered_configs
    
    # Collect rescalings for each configuration across all seeds
    config_rescalings = {}  # (width, n_hidden) -> list of rescalings across seeds
    
    for (width, n_hidden), base_dir_path in base_configs.items():
        rescalings_list = []
        for seed_val in seeds_to_use:
            # Discover configs for this seed to get the correct path
            seed_configs = discover_available_configs(results_dir, seed=seed_val, param_type=param_type, use_skips=use_skips)
            if (width, n_hidden) in seed_configs:
                seed_dir_path = seed_configs[(width, n_hidden)]
                rescaling_path = os.path.join(seed_dir_path, "loss_rescalings.npy")
                
                if os.path.exists(rescaling_path):
                    rescalings = _load_npy_safe(rescaling_path)
                    
                    if rescalings is not None and len(rescalings) > 0:
                        first_rescaling = rescalings[0]
                        
                        if isinstance(first_rescaling, np.ndarray):
                            if first_rescaling.ndim == 0:
                                first_rescaling = float(first_rescaling.item())
                            elif output_dim > 1:
                                first_rescaling = float(np.linalg.norm(first_rescaling, ord=2))
                            else:
                                first_rescaling = float(first_rescaling.item() if first_rescaling.size == 1 else first_rescaling[0])
                        else:
                            first_rescaling = float(first_rescaling)
                        
                        rescalings_list.append(first_rescaling)
        
        if rescalings_list:
            config_rescalings[(width, n_hidden)] = rescalings_list
    
    # Average rescalings across seeds and create data points
    data_points = []
    widths = []
    depths = []
    rescalings_minus_one = []
    
    for (width, n_hidden), rescalings_list in config_rescalings.items():
        avg_rescaling = np.mean(rescalings_list)
        widths.append(width)
        L = (n_hidden + 1) if n_hidden is not None else 1
        depths.append(L)
        rescalings_minus_one.append(avg_rescaling - 1.0)
        data_points.append((width, L, avg_rescaling - 1.0))
    
    if not data_points:
        print("  Warning: No rescaling data available")
        return
    
    print(f"  Loaded {len(data_points)} data points")
    
    # Convert to arrays
    widths_arr = np.array(widths)
    depths_arr = np.array(depths)
    rescalings_arr = np.array(rescalings_minus_one)
    
    # Calculate theoretical values: (L-1)/(gamma_0^2*N) without skips, 1/(gamma_0^2*N) + (L-1)/(gamma_0^2*N) with skips
    if param_type == 'sp':
        print("  Warning: Cannot compute error heatmap for 'sp' param_type (no theory)")
        return
    
    if use_skips:
        theory_values = 1.0 / (gamma_0**2 * widths_arr) + (depths_arr - 1) / (gamma_0**2 * widths_arr)
    else:
        theory_values = (depths_arr - 1) / (gamma_0**2 * widths_arr)
    
    # Calculate absolute errors
    absolute_errors = np.abs(rescalings_arr - theory_values)
    
    print(f"  Absolute error range: {np.min(absolute_errors):.6e} - {np.max(absolute_errors):.6e}")
    
    # Create a grid for the heatmap
    unique_widths = np.sort(np.unique(widths_arr))
    unique_depths = np.sort(np.unique(depths_arr))
    
    # Create error grid with shape (num_widths, num_depths) - swapped dimensions
    # We'll transpose for display, so error_grid[i, j] where i=width_idx, j=depth_idx
    error_grid = np.full((len(unique_widths), len(unique_depths)), np.nan, dtype=float)
    
    # Map each data point to the grid
    for i, (width, depth, rescaling_minus_one) in enumerate(data_points):
        width_idx = np.where(unique_widths == width)[0][0]
        depth_idx = np.where(unique_depths == depth)[0][0]
        # Store as error_grid[width_idx, depth_idx], then transpose for display
        # After transpose: x-axis = width (N), y-axis = depth (L)
        error_grid[width_idx, depth_idx] = absolute_errors[i]
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap using imshow (better for discrete data points)
    # Mask NaN values for visualization
    # imshow displays: first dimension (rows) -> y-axis, second dimension (cols) -> x-axis
    # With origin='lower', row 0 is at bottom, so depth_idx=0 (smallest depth) is at bottom
    masked_error_grid = np.ma.masked_invalid(error_grid)
    
    # Create heatmap with linear scale, set vmin=0 for colorbar
    # error_grid has shape (num_widths, num_depths), so we transpose to get (num_depths, num_widths)
    # After transpose, imshow displays: rows -> y-axis (depth L), cols -> x-axis (width N)
    im = ax.imshow(masked_error_grid.T, aspect='auto', origin='lower', 
                   cmap='viridis', interpolation='nearest', vmin=0)
    
    # Set ticks and labels with explicit font sizes
    # X-axis: widths
    num_widths = len(unique_widths)
    width_tick_positions = np.arange(num_widths)
    width_tick_labels = [f"${w}$" for w in unique_widths]
    ax.set_xticks(width_tick_positions)
    ax.set_xticklabels(width_tick_labels, fontsize=35)
    
    # Y-axis: depths - use actual values
    num_depths = len(unique_depths)
    depth_tick_positions = np.arange(num_depths)
    depth_tick_labels = [f"${d}$" for d in unique_depths]
    ax.set_yticks(depth_tick_positions)
    ax.set_yticklabels(depth_tick_labels, fontsize=35)
    
    # Set labels with increased font size
    ax.set_xlabel("Width $N$", fontsize=55, labelpad=LABEL_PAD)
    ax.set_ylabel("Depth $L$", fontsize=55, labelpad=LABEL_PAD)
    
    # Add colorbar with increased font sizes and padding
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$|\text{error}|$", 
                   fontsize=55, rotation=270, labelpad=80)
    cbar.ax.tick_params(labelsize=40)
    
    # Add text annotations for actual error values (optional, can be commented out if too cluttered)
    # for i in range(num_depths):
    #     for j in range(num_widths):
    #         if not np.isnan(error_grid[i, j]):
    #             text = ax.text(j, i, f'{error_grid[i, j]:.2e}',
    #                           ha="center", va="center", color="black", fontsize=8)
    
    # Set tick parameters with increased font size
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    
    # Remove all borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Save plot
    os.makedirs(plot_dir, exist_ok=True)
    filename = "rescaling_vs_width_depth_error_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', pad_inches=0.9)
    plt.close()
    
    print(f"  Saved error heatmap to {os.path.join(plot_dir, filename)}")


def plot_energy_loss_surfaces(results_dir, plot_dir, seeds=[0, 1, 2], param_type=None, 
                               widths_filter=None, n_hiddens_filter=None, use_skips=None):
    """Plot BP loss and PC energy as surfaces in 3D as functions of N and L.
    
    Creates a 3D plot with:
    - Width (N) on x-axis (log2 scale)
    - Depth L (number of layers) on y-axis (log2 scale)
    - Loss/Energy $l(\boldsymbol{\theta})$ on z-axis (log10 scale)
    - BP loss surface in red
    - PC energy surface in blue
    
    Averages over multiple seeds if available.
    """
    # Find all available seeds
    available_seeds = set()
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    # Build directory map
    dir_map = {}
    for root, _, files in os.walk(results_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            dir_map[root] = npy_files
    
    # Discover all available configurations
    base_configs = discover_available_configs(results_dir, seed=0, param_type=param_type, use_skips=use_skips)
    
    if not base_configs:
        print("  Warning: No configurations found")
        return
    
    # Filter configurations if filters are provided
    if widths_filter is not None and len(widths_filter) > 0:
        widths_filter_set = set(widths_filter)
        base_configs = {(w, h): path for (w, h), path in base_configs.items() if w in widths_filter_set}
    
    if n_hiddens_filter is not None and len(n_hiddens_filter) > 0:
        n_hiddens_filter_set = set(n_hiddens_filter)
        filtered_configs = {}
        for (w, n_hidden), path in base_configs.items():
            n_hidden_val = n_hidden if n_hidden is not None else 0
            if n_hidden_val in n_hiddens_filter_set:
                filtered_configs[(w, n_hidden)] = path
        base_configs = filtered_configs
    
    # Find all available seeds
    for root, _, files in os.walk(results_dir):
        if "energies.npy" in files or "losses.npy" in files:
            last_dir = root.split(os.sep)[-1]
            try:
                seed_val = int(last_dir)
                if param_type_str is None or param_type_str in root:
                    if use_skips_str is None or use_skips_str in root:
                        available_seeds.add(seed_val)
            except ValueError:
                pass
    
    if not available_seeds:
        print("  Warning: No seeds found")
        return
    
    seeds_to_use = sorted(available_seeds)
    print(f"  Found seeds: {seeds_to_use}, averaging over {len(seeds_to_use)} seeds")
    
    # Collect BP losses and PC energies for each configuration across all seeds
    config_bp_losses = {}  # (width, n_hidden) -> list of losses across seeds
    config_pc_energies = {}  # (width, n_hidden) -> list of energies across seeds
    
    for (width, n_hidden), base_dir_path in base_configs.items():
        bp_losses_list = []
        pc_energies_list = []
        width_str = f"{width}_width"
        n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden is not None else None
        
        for seed_val in seeds_to_use:
            seed_str = str(seed_val)
            pc_dir = None
            bp_dir = None
            
            # Find PC and BP directories for this configuration
            for dir_path, files in dir_map.items():
                if width_str in dir_path and dir_path.split(os.sep)[-1] == seed_str:
                    if n_hidden_str is None or n_hidden_str in dir_path:
                        if param_type_str is None or param_type_str in dir_path:
                            if use_skips_str is None or use_skips_str in dir_path:
                                if "energies.npy" in files:
                                    pc_dir = dir_path
                                elif "losses.npy" in files and "infer_mode" not in dir_path:
                                    bp_dir = dir_path
            
            # Load PC energy
            if pc_dir:
                energy_path = os.path.join(pc_dir, "energies.npy")
                if os.path.exists(energy_path):
                    energies = _load_npy_safe(energy_path)
                    if energies is not None and len(energies) > 0:
                        first_energy = energies[0]
                        if isinstance(first_energy, np.ndarray):
                            first_energy = float(first_energy.item() if first_energy.ndim == 0 else first_energy[0])
                        else:
                            first_energy = float(first_energy)
                        pc_energies_list.append(first_energy)
            
            # Load BP loss
            if bp_dir:
                loss_path = os.path.join(bp_dir, "losses.npy")
                if os.path.exists(loss_path):
                    losses = _load_npy_safe(loss_path)
                    if losses is not None and len(losses) > 0:
                        first_loss = losses[0]
                        if isinstance(first_loss, np.ndarray):
                            first_loss = float(first_loss.item() if first_loss.ndim == 0 else first_loss[0])
                        else:
                            first_loss = float(first_loss)
                        bp_losses_list.append(first_loss)
        
        if bp_losses_list:
            config_bp_losses[(width, n_hidden)] = bp_losses_list
        if pc_energies_list:
            config_pc_energies[(width, n_hidden)] = pc_energies_list
    
    # Get common configurations (those with both BP and PC data)
    common_configs = set(config_bp_losses.keys()) & set(config_pc_energies.keys())
    
    if not common_configs:
        print("  Warning: No configurations with both BP loss and PC energy data")
        return
    
    # Average across seeds and create data points
    bp_data_points = []
    pc_data_points = []
    widths = []
    depths = []
    bp_losses = []
    pc_energies = []
    
    for (width, n_hidden) in sorted(common_configs):
        avg_bp_loss = np.mean(config_bp_losses[(width, n_hidden)])
        avg_pc_energy = np.mean(config_pc_energies[(width, n_hidden)])
        
        widths.append(width)
        L = (n_hidden + 1) if n_hidden is not None else 1
        depths.append(L)
        bp_losses.append(avg_bp_loss)
        pc_energies.append(avg_pc_energy)
        bp_data_points.append((width, L, avg_bp_loss))
        pc_data_points.append((width, L, avg_pc_energy))
    
    if not bp_data_points or not pc_data_points:
        print("  Warning: No data points available")
        return
    
    print(f"  Loaded {len(bp_data_points)} data points")
    print(f"  BP loss range: {np.min(bp_losses):.6e} - {np.max(bp_losses):.6e}")
    print(f"  PC energy range: {np.min(pc_energies):.6e} - {np.max(pc_energies):.6e}")
    
    # Convert to arrays
    widths_arr = np.array(widths)
    depths_arr = np.array(depths)
    bp_losses_arr = np.array(bp_losses)
    pc_energies_arr = np.array(pc_energies)
    
    # Use log base 2 for both width and L
    log2_widths = np.log2(widths_arr)
    log2_depths = np.log2(depths_arr)
    
    # Use linear scale for z-axis (loss/energy)
    # No offset needed for linear scale
    
    # Create 3D plot
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mapping from (width, depth) to value
    bp_loss_dict = {(w, d): v for w, d, v in bp_data_points}
    pc_energy_dict = {(w, d): v for w, d, v in pc_data_points}
    
    # Get unique widths and depths
    unique_widths = np.sort(np.unique(widths_arr))
    unique_depths = np.sort(np.unique(depths_arr))
    
    # Create meshgrid from unique values
    W_mesh, L_mesh = np.meshgrid(unique_widths, unique_depths)
    
    # Map values to grid (use linear scale)
    bp_loss_grid = np.full_like(W_mesh, np.nan, dtype=float)
    pc_energy_grid = np.full_like(W_mesh, np.nan, dtype=float)
    
    for i, depth in enumerate(unique_depths):
        for j, width in enumerate(unique_widths):
            if (width, depth) in bp_loss_dict:
                bp_loss_grid[i, j] = bp_loss_dict[(width, depth)]
            if (width, depth) in pc_energy_dict:
                pc_energy_grid[i, j] = pc_energy_dict[(width, depth)]
    
    # Convert meshgrid to log2 scale for plotting
    W_log2_mesh = np.log2(W_mesh)
    L_log2_mesh = np.log2(L_mesh)
    
    # Plot PC energy surface in blue (plot first, lower zorder)
    try:
        surf_pc = ax.plot_surface(W_log2_mesh, L_log2_mesh, pc_energy_grid, 
                                  alpha=0.5, color='blue', shade=True, zorder=1, label='PC Energy')
    except (TypeError, AttributeError):
        try:
            surf_pc = ax.plot_surface(W_log2_mesh, L_log2_mesh, pc_energy_grid, 
                                     alpha=0.5, color='blue', shading='auto', zorder=1, label='PC Energy')
        except (TypeError, AttributeError):
            surf_pc = ax.plot_surface(W_log2_mesh, L_log2_mesh, pc_energy_grid, 
                                     alpha=0.5, color='blue', zorder=1, label='PC Energy')
    
    # Plot BP loss surface in red (plot second, higher zorder)
    try:
        surf_bp = ax.plot_surface(W_log2_mesh, L_log2_mesh, bp_loss_grid, 
                                   alpha=0.5, color='red', shade=True, zorder=2, label='BP Loss')
    except (TypeError, AttributeError):
        try:
            surf_bp = ax.plot_surface(W_log2_mesh, L_log2_mesh, bp_loss_grid, 
                                     alpha=0.5, color='red', shading='auto', zorder=2, label='BP Loss')
        except (TypeError, AttributeError):
            surf_bp = ax.plot_surface(W_log2_mesh, L_log2_mesh, bp_loss_grid, 
                                     alpha=0.5, color='red', zorder=2, label='BP Loss')
    
    # Set labels
    ax.set_xlabel("$N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_ylabel("$L$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_zlabel(r"$l(\boldsymbol{\theta})$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    
    # Set custom tick labels for x-axis (width)
    num_depths = len(set(depths_arr))
    min_width_log2 = np.log2(min(widths_arr))
    max_width_log2 = np.log2(max(widths_arr))
    width_tick_positions = np.linspace(min_width_log2, max_width_log2, num_depths)
    width_tick_positions = np.round(width_tick_positions).astype(int)
    width_tick_labels = [f"$2^{{{p}}}$" for p in width_tick_positions]
    ax.set_xticks(width_tick_positions)
    ax.set_xticklabels(width_tick_labels)
    
    # Set custom tick labels for y-axis (L)
    min_depth_log2 = np.log2(min(depths_arr))
    max_depth_log2 = np.log2(max(depths_arr))
    min_power = int(np.ceil(min_depth_log2))
    max_power = int(np.floor(max_depth_log2))
    depth_tick_positions = np.arange(min_power, max_power + 1)
    depth_tick_labels = [f"$2^{{{p}}}$" for p in depth_tick_positions]
    ax.set_yticks(depth_tick_positions)
    ax.set_yticklabels(depth_tick_labels)
    
    # Set custom tick labels for z-axis (loss/energy) - linear scale
    min_z = min(np.nanmin(bp_loss_grid), np.nanmin(pc_energy_grid))
    max_z = max(np.nanmax(bp_loss_grid), np.nanmax(pc_energy_grid))
    z_tick_positions = np.linspace(min_z, max_z, 5)
    # Format tick labels with appropriate precision
    z_tick_labels = []
    for p in z_tick_positions:
        if abs(p) < 0.01 or abs(p) > 1000:
            # Use scientific notation for very small or large values
            z_tick_labels.append(f"${p:.2e}$")
        else:
            # Use regular notation for moderate values
            z_tick_labels.append(f"${p:.2f}$")
    ax.set_zticks(z_tick_positions)
    ax.set_zticklabels(z_tick_labels)
    
    # Set tick parameters
    ax.tick_params(axis='x', labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis='y', labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis='z', labelsize=FONT_SIZES["tick"], pad=10)
    
    # Flip x-axis and invert L (y-axis)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Add legend (PC first, with star on F)
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor='blue', alpha=0.5, label=r'$\mathcal{F}^*(\boldsymbol{\theta})$ (PC)'),
        Patch(facecolor='red', alpha=0.5, label=r'$\mathcal{L}(\boldsymbol{\theta})$ (BP)')
    ]
    ax.legend(handles=handles, fontsize=FONT_SIZES["legend"], loc='upper right')
    
    # Save plot
    os.makedirs(plot_dir, exist_ok=True)
    filename = "energy_loss_surfaces.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', pad_inches=0.9)
    plt.close()
    
    print(f"  Saved energy/loss surfaces plot to {os.path.join(plot_dir, filename)}")


def plot_grad_cosine_similarity_vs_width_depth(results_dir, plot_dir, seeds=[0, 1, 2], 
                                                param_type=None, widths_filter=None, 
                                                n_hiddens_filter=None, use_skips=None):
    """Plot gradient cosine similarity as a function of width N and depth L in 3D.
    
    Creates a separate 3D plot for each seed in the provided list with:
    - Width (N) on x-axis (log2 scale)
    - Depth L (number of layers = n_hidden + 1) on y-axis (log2 scale)
    - Cosine similarity on z-axis (linear scale)
    - Data points in blue
    
    Args:
        seeds: List of seed values to plot. One plot will be generated for each seed.
        n_hiddens_filter: Filter by n_hidden (number of hidden layers). L = n_hidden + 1 is computed during data loading.
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    """
    if not seeds:
        print("  Warning: No seeds provided")
        return
    
    seeds_to_plot = sorted(seeds)
    print(f"  Creating plots for seeds: {seeds_to_plot} ({len(seeds_to_plot)} plots)")
    
    # Create a plot for each seed
    for seed_val in seeds_to_plot:
        print(f"\n  Generating plot for seed {seed_val}...")
        
        # Discover all available configurations
        configs = discover_available_configs(results_dir, seed=seed_val, param_type=param_type, use_skips=use_skips)
        
        if not configs:
            print(f"  Warning: No configurations found for seed {seed_val}")
            continue
        
        print(f"  Found {len(configs)} configurations for seed {seed_val}")
        
        # Filter configurations if filters are provided
        if widths_filter is not None and len(widths_filter) > 0:
            widths_filter_set = set(widths_filter)
            configs = {(w, h): path for (w, h), path in configs.items() if w in widths_filter_set}
        
        if n_hiddens_filter is not None and len(n_hiddens_filter) > 0:
            n_hiddens_filter_set = set(n_hiddens_filter)
            filtered_configs = {}
            for (w, n_hidden), path in configs.items():
                n_hidden_val = n_hidden if n_hidden is not None else 0
                if n_hidden_val in n_hiddens_filter_set:
                    filtered_configs[(w, n_hidden)] = path
            configs = filtered_configs
        
        # Load cosine similarity data for each configuration
        data_points = []
        widths = []
        depths = []
        cosine_similarities = []
        
        for (width, n_hidden), dir_path in configs.items():
            cosine_sim_path = os.path.join(dir_path, "grad_cosine_similarities.npy")
            
            if os.path.exists(cosine_sim_path):
                cosine_sims = _load_npy_safe(cosine_sim_path)
                
                if cosine_sims is not None and len(cosine_sims) > 0:
                    # Use first value
                    first_cosine_sim = cosine_sims[0]
                    
                    # Convert to float, handling numpy scalars
                    if isinstance(first_cosine_sim, np.ndarray):
                        first_cosine_sim = float(first_cosine_sim.item() if first_cosine_sim.ndim == 0 else first_cosine_sim[0])
                    else:
                        first_cosine_sim = float(first_cosine_sim)
                    
                    # Filter out negative cosine similarities
                    if first_cosine_sim >= 0:
                        widths.append(width)
                        L = (n_hidden + 1) if n_hidden is not None else 1
                        depths.append(L)
                        cosine_similarities.append(first_cosine_sim)
                        data_points.append((width, L, first_cosine_sim))
        
        if not data_points:
            print("  Warning: No cosine similarity data available")
            continue
        
        print(f"  Loaded {len(data_points)} data points")
        if len(data_points) > 0:
            print(f"  Width range: {min(widths)} - {max(widths)}")
            print(f"  L (layers) range: {min(depths)} - {max(depths)}")
            print(f"  Cosine similarity range: {min(cosine_similarities):.6f} - {max(cosine_similarities):.6f}")
        
        # Create 3D plot
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to arrays
        widths_arr = np.array(widths)
        depths_arr = np.array(depths)
        cosine_sims_arr = np.array(cosine_similarities)
        
        # Use log base 2 for both width and L
        log2_widths = np.log2(widths_arr)
        log2_depths = np.log2(depths_arr)
        
        # Use linear scale for z-axis (cosine similarity)
        # Cosine similarity is in [-1, 1], but typically close to 1
        
        # Plot data points in blue
        ax.scatter(log2_widths, log2_depths, cosine_sims_arr, 
                   color='#1E88E5', s=200, alpha=0.9, label='Data', depthshade=True, 
                   edgecolors='darkblue', linewidths=1)
        
        # Set labels
        ax.set_xlabel("$N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.set_ylabel("$L$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.set_zlabel(r"$\cos(\nabla_{\boldsymbol{\theta}_t}\mathcal{F}^*, \nabla_{\boldsymbol{\theta}_t}\mathcal{L})$", 
                      fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        
        # Set custom tick labels for x-axis (width)
        num_depths = len(set(depths_arr))
        min_width_log2 = np.log2(min(widths_arr))
        max_width_log2 = np.log2(max(widths_arr))
        width_tick_positions = np.linspace(min_width_log2, max_width_log2, num_depths)
        width_tick_positions = np.round(width_tick_positions).astype(int)
        width_tick_labels = [f"$2^{{{p}}}$" for p in width_tick_positions]
        ax.set_xticks(width_tick_positions)
        ax.set_xticklabels(width_tick_labels)
        
        # Set custom tick labels for y-axis (L)
        min_depth_log2 = np.log2(min(depths_arr))
        max_depth_log2 = np.log2(max(depths_arr))
        min_power = int(np.ceil(min_depth_log2))
        max_power = int(np.floor(max_depth_log2))
        depth_tick_positions = np.arange(min_power, max_power + 1)
        depth_tick_labels = [f"$2^{{{p}}}$" for p in depth_tick_positions]
        ax.set_yticks(depth_tick_positions)
        ax.set_yticklabels(depth_tick_labels)
        
        # Set z-axis limits to [0, 1] for cosine similarity
        ax.set_zlim(0, 1)
        
        # Set custom tick labels for z-axis (cosine similarity) - linear scale
        # Create 5 evenly spaced ticks from 0 to 1
        z_tick_positions = np.linspace(0, 1, 5)
        z_tick_labels = [f"${p:.2f}$" for p in z_tick_positions]
        ax.set_zticks(z_tick_positions)
        ax.set_zticklabels(z_tick_labels)
        
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=FONT_SIZES["tick"])
        ax.tick_params(axis='y', labelsize=FONT_SIZES["tick"])
        ax.tick_params(axis='z', labelsize=FONT_SIZES["tick"], pad=10)
        
        # Flip x-axis only
        ax.invert_xaxis()
        
        # Save plot
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"grad_cosine_similarity_vs_width_depth_seed_{seed_val}.pdf"
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', pad_inches=0.9)
        plt.close()
        
        print(f"  Saved plot to {os.path.join(plot_dir, filename)}")


def plot_grad_cosine_similarity_heatmap(results_dir, plot_dir, seed=0, 
                                        param_type=None, widths_filter=None, 
                                        n_hiddens_filter=None, use_skips=None,
                                        time_step=0):
    """Plot a heatmap showing cosine similarity between BP and PC gradients as a function of N and L.
    
    Creates a 2D heatmap with:
    - Width (N) on x-axis
    - Depth L (number of layers) on y-axis
    - Color representing cosine similarity at the specified time step
    
    Averages over multiple seeds (0, 1, 2) if available.
    
    Args:
        time_step: Time step index to use (0 for first, 50 for middle, 100 for last, etc.)
    """
    print(f"  Plotting cosine similarity heatmap at time step t={time_step}")
    
    # Find all available seeds
    available_seeds = set()
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    for root, _, files in os.walk(results_dir):
        if "grad_cosine_similarities.npy" in files:
            last_dir = root.split(os.sep)[-1]
            try:
                seed_val = int(last_dir)
                # Filter by param_type and use_skips if specified
                if param_type_str is None or param_type_str in root:
                    if use_skips_str is None or use_skips_str in root:
                        available_seeds.add(seed_val)
            except ValueError:
                pass
    
    if not available_seeds:
        print("  Warning: No seeds found")
        return
    
    seeds_to_use = sorted(available_seeds)
    print(f"  Found seeds: {seeds_to_use}, averaging over {len(seeds_to_use)} seeds")
    
    # Discover all available configurations (using first seed to get structure)
    print("  Discovering available (width, n_hidden) configurations...")
    base_configs = discover_available_configs(results_dir, seed=seeds_to_use[0], param_type=param_type, use_skips=use_skips)
    
    if not base_configs:
        print("  Warning: No configurations found")
        return
    
    # Filter configurations if filters are provided
    if widths_filter is not None and len(widths_filter) > 0:
        widths_filter_set = set(widths_filter)
        base_configs = {(w, h): path for (w, h), path in base_configs.items() if w in widths_filter_set}
    
    if n_hiddens_filter is not None and len(n_hiddens_filter) > 0:
        n_hiddens_filter_set = set(n_hiddens_filter)
        filtered_configs = {}
        for (w, n_hidden), path in base_configs.items():
            n_hidden_val = n_hidden if n_hidden is not None else 0
            if n_hidden_val in n_hiddens_filter_set:
                filtered_configs[(w, n_hidden)] = path
        base_configs = filtered_configs
    
    # Collect cosine similarities for each configuration across all seeds
    config_cosine_sims = {}  # (width, n_hidden) -> list of cosine similarities across seeds
    
    for (width, n_hidden), base_dir_path in base_configs.items():
        cosine_sims_list = []
        for seed_val in seeds_to_use:
            # Discover configs for this seed to get the correct path
            seed_configs = discover_available_configs(results_dir, seed=seed_val, param_type=param_type, use_skips=use_skips)
            if (width, n_hidden) in seed_configs:
                seed_dir_path = seed_configs[(width, n_hidden)]
                cosine_sim_path = os.path.join(seed_dir_path, "grad_cosine_similarities.npy")
                
                if os.path.exists(cosine_sim_path):
                    cosine_sims = _load_npy_safe(cosine_sim_path)
                    
                    if cosine_sims is not None and len(cosine_sims) > 0:
                        # Use the specified time step, clamping to available range
                        idx = min(time_step, len(cosine_sims) - 1)
                        cosine_sim_at_step = cosine_sims[idx]
                        
                        if isinstance(cosine_sim_at_step, np.ndarray):
                            cosine_sim_at_step = float(cosine_sim_at_step.item() if cosine_sim_at_step.ndim == 0 else cosine_sim_at_step[0])
                        else:
                            cosine_sim_at_step = float(cosine_sim_at_step)
                        
                        # Only add non-negative cosine similarities
                        if cosine_sim_at_step >= 0:
                            cosine_sims_list.append(cosine_sim_at_step)
        
        if cosine_sims_list:
            config_cosine_sims[(width, n_hidden)] = cosine_sims_list
    
    # Average cosine similarities across seeds and create data points
    data_points = []
    widths = []
    depths = []
    cosine_similarities = []
    
    for (width, n_hidden), cosine_sims_list in config_cosine_sims.items():
        # cosine_sims_list already contains only non-negative values
        if cosine_sims_list:
            avg_cosine_sim = np.mean(cosine_sims_list)
            widths.append(width)
            L = (n_hidden + 1) if n_hidden is not None else 1
            depths.append(L)
            cosine_similarities.append(avg_cosine_sim)
            data_points.append((width, L, avg_cosine_sim))
    
    if not data_points:
        print("  Warning: No cosine similarity data available")
        return
    
    print(f"  Loaded {len(data_points)} data points")
    print(f"  Cosine similarity range: {np.min(cosine_similarities):.6f} - {np.max(cosine_similarities):.6f}")
    
    # Convert to arrays
    widths_arr = np.array(widths)
    depths_arr = np.array(depths)
    cosine_sims_arr = np.array(cosine_similarities)
    
    # Create a grid for the heatmap
    unique_widths = np.sort(np.unique(widths_arr))
    unique_depths = np.sort(np.unique(depths_arr))
    
    # Create cosine similarity grid with shape (num_widths, num_depths)
    cosine_grid = np.full((len(unique_widths), len(unique_depths)), np.nan, dtype=float)
    
    # Map each data point to the grid
    for i, (width, depth, cosine_sim) in enumerate(data_points):
        width_idx = np.where(unique_widths == width)[0][0]
        depth_idx = np.where(unique_depths == depth)[0][0]
        cosine_grid[width_idx, depth_idx] = cosine_sim
    
    # Fill NaN values with 0
    cosine_grid = np.nan_to_num(cosine_grid, nan=0.0)
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with linear scale, inverted colormap
    # cosine_grid has shape (num_widths, num_depths), so we transpose to get (num_depths, num_widths)
    # Explicitly set vmin and vmax to ensure consistent colorbar across all time steps
    im = ax.imshow(cosine_grid.T, aspect='auto', origin='lower', 
                   cmap='viridis_r', interpolation='nearest', vmin=0, vmax=1)
    
    # Add title with increased font size and padding
    ax.set_title(f"$t={time_step}$", fontsize=55, pad=20)
    
    # Set ticks and labels with explicit font sizes
    # X-axis: widths
    num_widths = len(unique_widths)
    width_tick_positions = np.arange(num_widths)
    width_tick_labels = [f"${w}$" for w in unique_widths]
    ax.set_xticks(width_tick_positions)
    ax.set_xticklabels(width_tick_labels, fontsize=35)
    
    # Y-axis: depths
    num_depths = len(unique_depths)
    depth_tick_positions = np.arange(num_depths)
    depth_tick_labels = [f"${d}$" for d in unique_depths]
    ax.set_yticks(depth_tick_positions)
    ax.set_yticklabels(depth_tick_labels, fontsize=35)
    
    # Set labels with increased font size
    ax.set_xlabel("Width $N$", fontsize=55, labelpad=LABEL_PAD)
    ax.set_ylabel("Depth $L$", fontsize=55, labelpad=LABEL_PAD)
    
    # Add colorbar with increased font sizes and padding
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\cos(\nabla_{\boldsymbol{\theta}_t}\mathcal{F}^*, \nabla_{\boldsymbol{\theta}_t}\mathcal{L})$", 
                   fontsize=55, rotation=270, labelpad=80)
    cbar.ax.tick_params(labelsize=40)
    
    # Set tick parameters with increased font size (redundant but ensures consistency)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    
    # Remove all borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Save plot
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"grad_cosine_similarity_heatmap_t{time_step}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', pad_inches=0.9, dpi=300)
    plt.close()
    
    print(f"  Saved cosine similarity heatmap to {os.path.join(plot_dir, filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rescaling vs width and depth in 3D")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory containing experiment outputs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help="List of seed values to plot. One plot will be generated for each seed. (default: [0, 1, 2])"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="width_vs_depth_plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--gamma_0",
        type=float,
        default=None,
        help="Gamma_0 value for theoretical plane (default: extracted from data directory)"
    )
    parser.add_argument(
        "--param_type",
        type=str,
        choices=["sp", "mupc"],
        default=None,
        help="Parameter type to filter by ('sp' or 'mupc', None to load all)"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=1,
        help="Output dimension. If > 1, labels will use spectral norm notation (default: 1)"
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs='+',
        default=[2, 8, 32, 128, 512, 2048],
        help="List of widths N to plot. If not provided, plots all available widths."
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        nargs='+',
        default=[1, 3, 7, 15, 31],
        help="List of n_hidden (number of hidden layers) to plot. L (number of layers) = n_hidden + 1. If not provided, plots all available n_hidden values."
    )
    
    def parse_use_skips(value):
        """Parse use_skips argument from string to bool or None."""
        if value is None:
            return None
        lower_val = str(value).lower()
        if lower_val in ('true', '1', 'yes'):
            return True
        elif lower_val in ('false', '0', 'no'):
            return False
        else:
            raise ValueError(f"Invalid use_skips value: {value}. Must be True/False, true/false, 1/0, or yes/no.")
    
    parser.add_argument(
        "--use_skips",
        type=parse_use_skips,
        default=None,
        help="Filter results by use_skips setting. Use --use_skips True/False or --use_skips true/false. If not provided, automatically plots both True and False in separate folders."
    )
    parser.add_argument(
        "--linear_z_scale",
        action="store_true",
        default=False,
        help="Use linear scale for z-axis of rescaling 3D scatter plot. If not provided, uses log10 scale (default)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=["Fashion-MNIST", "CIFAR10"],
        help="List of dataset names to plot. Plots will be saved in plot_dir/dataset/. (default: ['toy'])"
    )
    parser.add_argument(
        "--dataset_input_dims",
        type=int,
        nargs='+',
        default=[784, 3072],
        help="List of input dimensions corresponding to each dataset in --datasets. Must have same length as --datasets. If not provided, will use defaults: toy=40, MNIST=784, Fashion-MNIST=784, CIFAR10=3072. Results will be looked up in results_dir/{input_dim}_input_dim/..."
    )
    
    args = parser.parse_args()
    
    # Default input_dim mapping for common datasets
    default_input_dims = {
        "toy": 40,
        "MNIST": 784,
        "Fashion-MNIST": 784,
        "CIFAR10": 3072
    }
    
    # Determine input_dims for each dataset
    if args.dataset_input_dims is not None:
        if len(args.dataset_input_dims) != len(args.datasets):
            raise ValueError(
                f"Number of input_dims ({len(args.dataset_input_dims)}) must match "
                f"number of datasets ({len(args.datasets)})"
            )
        dataset_input_dims = dict(zip(args.datasets, args.dataset_input_dims))
    else:
        # Use defaults
        dataset_input_dims = {}
        for dataset in args.datasets:
            if dataset in default_input_dims:
                dataset_input_dims[dataset] = default_input_dims[dataset]
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. Please provide --dataset_input_dims "
                    f"or use one of: {list(default_input_dims.keys())}"
                )
    
    # Determine which use_skips values to process
    if args.use_skips is not None:
        use_skips_values = [args.use_skips]
    else:
        use_skips_values = [True, False]
    
    # Create base plot directory
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Process each dataset
    for dataset in args.datasets:
        print(f"\n{'#'*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'#'*80}\n")
        
        # Get input_dim for this dataset
        input_dim = dataset_input_dims[dataset]
        print(f"  Using input_dim: {input_dim}")
        
        # Construct results directory path using input_dim structure
        # Results are stored in: results_dir/{input_dim}_input_dim/...
        dataset_results_dir = os.path.join(args.results_dir, f"{input_dim}_input_dim")
        
        # Check if results directory exists
        if not os.path.exists(dataset_results_dir):
            print(f"  Warning: Results directory not found: {dataset_results_dir}")
            print(f"  Skipping dataset: {dataset}\n")
            continue
        
        # Create dataset-specific plot directory
        dataset_plot_dir = os.path.join(args.plot_dir, dataset)
        os.makedirs(dataset_plot_dir, exist_ok=True)
        
        # Process each use_skips configuration
        for use_skips in use_skips_values:
            print(f"\n{'='*80}")
            print(f"Processing use_skips = {use_skips}")
            print(f"{'='*80}\n")
            
            # Create subdirectory for this skip status within the dataset directory
            skip_status_dir = os.path.join(dataset_plot_dir, f"{use_skips}_use_skips")
            os.makedirs(skip_status_dir, exist_ok=True)
            
            # Generate 3D plot
            print("Generating 3D rescaling vs width and depth plot...")
            plot_rescaling_vs_width_depth(
                dataset_results_dir,
                skip_status_dir,
                seeds=args.seeds,
                gamma_0=args.gamma_0,
                param_type=args.param_type,
                output_dim=args.output_dim,
                widths_filter=args.widths,
                n_hiddens_filter=args.n_hiddens,
                use_skips=use_skips,
                linear_z_scale=args.linear_z_scale
            )
            
            # Generate error heatmap
            print("\nGenerating error heatmap...")
            plot_error_heatmap(
                dataset_results_dir,
                skip_status_dir,
                seed=args.seeds[0] if args.seeds else 0,  # Use first seed for error heatmap (which averages over all seeds anyway)
                gamma_0=args.gamma_0,
                param_type=args.param_type,
                output_dim=args.output_dim,
                widths_filter=args.widths,
                n_hiddens_filter=args.n_hiddens,
                use_skips=use_skips
            )
            
            # Generate energy/loss surfaces 3D plot
            print("\nGenerating energy/loss surfaces 3D plot...")
            plot_energy_loss_surfaces(
                dataset_results_dir,
                skip_status_dir,
                seeds=args.seeds,
                param_type=args.param_type,
                widths_filter=args.widths,
                n_hiddens_filter=args.n_hiddens,
                use_skips=use_skips
            )
            
            # Generate gradient cosine similarity 3D plot
            print("\nGenerating gradient cosine similarity 3D plot...")
            plot_grad_cosine_similarity_vs_width_depth(
                dataset_results_dir,
                skip_status_dir,
                seeds=args.seeds,
                param_type=args.param_type,
                widths_filter=args.widths,
                n_hiddens_filter=args.n_hiddens,
                use_skips=use_skips
            )
            
            # Generate gradient cosine similarity heatmaps at different time steps
            print("\nGenerating gradient cosine similarity heatmaps at different time steps...")
            for time_step in [0, 50, 100]:
                print(f"\n  Generating heatmap for t={time_step}...")
                plot_grad_cosine_similarity_heatmap(
                    dataset_results_dir,
                    skip_status_dir,
                    seed=args.seeds[0] if args.seeds else 0,
                    param_type=args.param_type,
                    widths_filter=args.widths,
                    n_hiddens_filter=args.n_hiddens,
                    use_skips=use_skips,
                    time_step=time_step
                )
            
            print(f"\nCompleted processing for use_skips = {use_skips}")
        
        print(f"\nCompleted processing for dataset: {dataset}")
    
    print("\nDone!")
