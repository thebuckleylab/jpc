import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
})

# Plot styling constants
FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7


def extract_dataset_id(results_dir):
    """Extract dataset ID from results directory structure.
    
    Looks for directories matching '*_input_dim' pattern in the results directory.
    Returns the dataset ID (e.g., '3072_input_dim') or 'unknown' if not found.
    """
    if not os.path.exists(results_dir):
        return 'unknown'
    
    # Look for directories matching '*_input_dim' pattern
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.endswith('_input_dim'):
            return item
    
    # If not found, return 'unknown'
    return 'unknown'


def is_sequential_colormap(colormap_name):
    """Check if a colormap is sequential."""
    sequential = {
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys',
        'YlOrRd', 'YlOrBr', 'YlGnBu', 'YlGn', 'RdPu',
        'BuGn', 'BuPu', 'GnBu', 'PuBu', 'PuBuGn', 'PuRd', 'OrRd',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'cool', 'hot',
        'copper', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter'
    }
    return colormap_name in sequential


def get_color_val(idx, n_widths, colormap_name):
    """Get color value for colormap."""
    if is_sequential_colormap(colormap_name):
        return 0.15 + (idx / max(n_widths - 1, 1)) * 0.85 if n_widths > 1 else 0.15
    return (idx / max(n_widths - 1, 1)) * 0.85 if n_widths > 1 else 0


def setup_plot(xlabel, ylabel, log_scale=False):
    """Setup common plot styling."""
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(xlabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(ylabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    # Only create legend if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=FONT_SIZES["legend"])
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    if log_scale:
        plt.yscale('log', base=10)


def save_plot(results_dir, filename, n_hidden=None, add_suffix=True):
    """Save plot with optional n_hidden suffix.
    
    Args:
        results_dir: Directory to save the plot
        filename: Base filename
        n_hidden: Number of hidden layers (for suffix)
        add_suffix: If True and n_hidden is not None, adds _H{n_hidden} suffix
    """
    if n_hidden is not None and add_suffix:
        filename = filename.replace('.pdf', f'_H{n_hidden}.pdf')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight')
    plt.close()


def plot_widths_series(data, data_key, results_dir, colormap_name, ylabel, filename, 
                       n_hidden=None, log_scale=False, log_xaxis=False, label_prefix='$N = {}$'):
    """Generic function to plot a series over widths."""
    plt.figure(figsize=FIG_SIZE)
    widths_list = sorted([w for w in data["widths"] if w in data[data_key]])
    
    if widths_list:
        colormap = plt.get_cmap(colormap_name)
        n_widths = len(widths_list)
        
        for idx, width in enumerate(widths_list):
            values = np.array(data[data_key][width]).flatten()
            iterations = np.arange(1, len(values) + 1)
            color = colormap(get_color_val(idx, n_widths, colormap_name))
            plt.plot(iterations, values, label=label_prefix.format(width),
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    setup_plot("$t$", ylabel, log_scale)
    if log_xaxis:
        plt.xscale('log', base=10)
    save_plot(results_dir, filename, n_hidden)


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


def discover_available_widths(results_dir, seed=0, n_hidden=None, required_files=None, param_type=None, use_skips=None):
    """Discover all available widths from the results directory structure.
    
    Args:
        results_dir: Directory to search
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        required_files: List of file names that must be present. If None, defaults to ["loss_rescalings.npy"]
        param_type: Parameter type to filter by ('sp' or 'mupc', None to load all)
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    """
    if required_files is None:
        required_files = ["loss_rescalings.npy"]
    
    widths = set()
    seed_str = str(seed)
    n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    for root, _, files in os.walk(results_dir):
        # Check if any of the required files are present
        if any(req_file in files for req_file in required_files):
            # Check if this directory matches our seed and n_hidden criteria
            if root.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in root:
                    # Filter by param_type if specified
                    if param_type_str is None or param_type_str in root:
                        # Filter by use_skips if specified
                        if use_skips_str is None or use_skips_str in root:
                            # Extract width from directory path
                            for part in root.split(os.sep):
                                if part.endswith("_width"):
                                    try:
                                        width = int(part.replace("_width", ""))
                                        widths.add(width)
                                        break
                                    except ValueError:
                                        pass
    
    return sorted(list(widths))


def load_data_from_dir(results_dir, widths, seed=0, n_hidden=None, param_type=None, use_skips=None):
    """Load all data from the results directory structure.
    
    Args:
        results_dir: Base results directory
        widths: List of widths to load
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        param_type: Parameter type to filter by ('sp' or 'mupc', None to load all)
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    """
    data = {
        "widths": widths, "pc_rescalings": {}, "pc_energies": {},
        "pc_train_losses": {}, "pc_grads": {}, "bp_losses": {},
        "bp_grads": {}, "grad_cosine_similarities": {}, "dmft_loss": None, "gamma_0": None
    }
    
    # Check for Adam optimizer
    use_adam = any("adam_param_optim" in root or "adam_optim_id" in root 
                   for root, _, _ in os.walk(results_dir))
    
    # Load DMFT loss
    if not use_adam:
        # Extract dataset ID from results directory
        dataset_id = extract_dataset_id(results_dir)
        
        # First, try to find DMFT loss file by searching for the actual pattern used in train.py
        # Pattern: dmft_loss_{gamma_0}_gamma_0_seed_{seed}.npy
        seed_str = str(seed)
        dmft_found = False
        
        # Search for files matching the pattern (check results_dir directly and all subdirectories)
        search_locations = [results_dir]
        for root, _, files in os.walk(results_dir):
            if root != results_dir:
                search_locations.append(root)
        
        for search_dir in search_locations:
            if not os.path.isdir(search_dir):
                continue
            for file in os.listdir(search_dir):
                if file.startswith("dmft_loss_") and file.endswith(f"_seed_{seed_str}.npy"):
                    # Check if it matches the gamma_0 pattern: dmft_loss_{gamma_0}_gamma_0_seed_{seed}.npy
                    file_base = file.replace("dmft_loss_", "").replace(f"_seed_{seed_str}.npy", "")
                    if file_base.endswith("_gamma_0"):
                        # Extract gamma_0 value
                        gamma_0_str = file_base.replace("_gamma_0", "")
                        try:
                            gamma_0_val = float(gamma_0_str)
                            dmft = _load_npy_safe(os.path.join(search_dir, file))
                            if dmft is not None:
                                data["dmft_loss"] = dmft
                                data["gamma_0"] = gamma_0_val
                                dmft_found = True
                                break
                        except ValueError:
                            pass
            if dmft_found:
                break
        
        # If not found with gamma_0 pattern, try other patterns
        if not dmft_found:
            dmft_paths = []
            if n_hidden is not None:
                dmft_paths.extend([
                    f"dmft_loss_{dataset_id}_n_hidden_{n_hidden}_seed_{seed_str}.npy",
                    f"dmft_loss_n_hidden_{n_hidden}_seed_{seed_str}.npy",
                ])
            dmft_paths.append(f"dmft_loss_seed_{seed_str}.npy")
            
            for path in dmft_paths:
                dmft = _load_npy_safe(os.path.join(results_dir, path))
                if dmft is not None:
                    data["dmft_loss"] = dmft
                    break
    
    # Build directory map
    dir_map = {}
    for root, _, files in os.walk(results_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            dir_map[root] = npy_files
    
    # Load data for each width
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    for width in widths:
        width_str, seed_str = f"{width}_width", str(seed)
        n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
        
        pc_dir = bp_dir = None
        for dir_path, files in dir_map.items():
            if width_str in dir_path and dir_path.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in dir_path:
                    # Filter by param_type if specified
                    if param_type_str is None or param_type_str in dir_path:
                        # Filter by use_skips if specified
                        if use_skips_str is None or use_skips_str in dir_path:
                            if "energies.npy" in files:
                                pc_dir = dir_path
                            elif "losses.npy" in files:
                                bp_dir = dir_path
        
        # Load PC data
        if pc_dir:
            if data["gamma_0"] is None:
                for part in pc_dir.split(os.sep):
                    if part.endswith("_gamma_0"):
                        try:
                            data["gamma_0"] = float(part.replace("_gamma_0", ""))
                            break
                        except ValueError:
                            pass
            
            pc_files = {
                "pc_energies": "energies.npy",
                "pc_train_losses": "train_losses.npy",
                "pc_rescalings": "loss_rescalings.npy"
            }
            for key, filename in pc_files.items():
                arr = _load_npy_safe(os.path.join(pc_dir, filename))
                if arr is not None:
                    data[key][width] = arr
            
            grads = _load_npy_safe(os.path.join(pc_dir, "grads.npy"), flatten=False)
            if grads is not None:
                data["pc_grads"][width] = grads
            
            # Load cosine similarities if available (preferred over computing from gradients)
            cosine_sims = _load_npy_safe(os.path.join(pc_dir, "grad_cosine_similarities.npy"))
            if cosine_sims is not None:
                data["grad_cosine_similarities"][width] = cosine_sims
        
        # Load BP data
        if bp_dir:
            for key, filename in [("bp_losses", "losses.npy"), ("bp_grads", "bp_grads.npy")]:
                arr = _load_npy_safe(os.path.join(bp_dir, filename), flatten=(key == "bp_losses"))
                if arr is not None:
                    data[key][width] = arr
    
    return data


def plot_rescalings(data, results_dir, colormap_name='viridis', n_hidden=None, output_dim=1, log_xaxis=False):
    """Plot PC rescalings S during training as a function of different widths N."""
    if output_dim > 1:
        ylabel = r"$\|\mathbf{S}(\boldsymbol{\theta}_t)\|_2$"
    else:
        ylabel = r"$s(\boldsymbol{\theta}_t)$"
    plot_widths_series(data, "pc_rescalings", results_dir, colormap_name,
                       ylabel, "pc_rescaling.pdf", n_hidden, log_scale=False, log_xaxis=log_xaxis)


def plot_rescaling_vs_width(data, plot_dir, colormap_name='viridis', n_hidden=None, gamma_0=None, 
                            seed=0, use_all_widths=False, data_results_dir=None, output_dim=1, param_type=None, use_skips=None):
    """Plot PC rescaling - 1 as a function of width N at the first training step, with theoretical L/(gamma_0^2*N) line where L = n_hidden + 1.
    
    If use_all_widths is True, discovers and loads all available widths from data_results_dir instead of using data["widths"].
    plot_dir is used for saving the plot.
    """
    # Get gamma_0 from data if not provided
    if gamma_0 is None:
        if data.get("gamma_0") is not None:
            gamma_0 = data["gamma_0"]
        else:
            print("  Warning: gamma_0 not found in data, using default 2.0")
            gamma_0 = 2.0
    
    # Calculate L = n_hidden + 1 (number of layers)
    if n_hidden is None:
        print("  Warning: n_hidden not provided, using default L=1 for theory line")
        L = 1
    else:
        L = n_hidden + 1
    
    # If use_all_widths is True, discover and load all available widths
    if use_all_widths:
        if data_results_dir is None:
            print("  Warning: use_all_widths=True but data_results_dir not provided, falling back to data widths")
            widths_to_use = data["widths"]
            rescalings_data = data["pc_rescalings"]
        else:
            print("  Discovering all available widths for rescaling_vs_width plot...")
            all_widths = discover_available_widths(data_results_dir, seed=seed, n_hidden=n_hidden, param_type=param_type, use_skips=use_skips)
            if all_widths:
                print(f"  Found {len(all_widths)} widths: {all_widths}")
                # Load rescaling data for all discovered widths
                widths_to_use = all_widths
                rescalings_data = {}
                seed_str = str(seed)
                n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
                param_type_str = f"{param_type}_param_type" if param_type else None
                use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
                
                for root, _, files in os.walk(data_results_dir):
                    if "loss_rescalings.npy" in files:
                        if root.split(os.sep)[-1] == seed_str:
                            if n_hidden_str is None or n_hidden_str in root:
                                # Filter by param_type if specified
                                if param_type_str is None or param_type_str in root:
                                    # Filter by use_skips if specified
                                    if use_skips_str is None or use_skips_str in root:
                                        # Extract width from directory path
                                        width = None
                                        for part in root.split(os.sep):
                                            if part.endswith("_width"):
                                                try:
                                                    width = int(part.replace("_width", ""))
                                                    break
                                                except ValueError:
                                                    pass
                                        
                                        if width is not None and width in all_widths:
                                            arr = _load_npy_safe(os.path.join(root, "loss_rescalings.npy"))
                                            if arr is not None:
                                                rescalings_data[width] = arr
            else:
                print("  Warning: No widths discovered, falling back to data widths")
                widths_to_use = data["widths"]
                rescalings_data = data["pc_rescalings"]
    else:
        widths_to_use = data["widths"]
        rescalings_data = data["pc_rescalings"]
    
    plt.figure(figsize=FIG_SIZE)
    
    # Extract first rescaling value for each width
    widths_list = sorted([w for w in widths_to_use if w in rescalings_data])
    first_rescalings_minus_one = []
    valid_widths = []
    
    missing_widths = sorted([w for w in widths_to_use if w not in rescalings_data])
    if missing_widths:
        print(f"  Warning: Missing rescaling data for widths: {missing_widths}")
        print(f"  (Rescalings are only saved when infer_mode='closed_form')")
    
    for width in widths_list:
        rescalings = np.array(rescalings_data[width]).flatten()
        if len(rescalings) > 0:
            first_rescalings_minus_one.append(rescalings[0] - 1.0)
            valid_widths.append(width)
        else:
            print(f"  Warning: Width {width} has empty rescaling data")
    
    if not valid_widths:
        print("  Warning: No rescaling data available")
        plt.close()
        return
    
    print(f"  Plotting {len(valid_widths)} data points for widths: {valid_widths}")
    
    # Plot data points
    plt.scatter(valid_widths, first_rescalings_minus_one, s=300, alpha=ALPHA,
                color='#1E88E5', label='Data', zorder=3)
    
    # Plot theoretical line
    if param_type != 'sp':
        min_width, max_width = min(valid_widths), max(valid_widths)
        theory_widths = np.logspace(np.log10(min_width), np.log10(max_width), 100)
        
        if use_skips == True:
            # Theory line for use_skips=True: (1 + L)/(gamma_0^2 * N)
            theory_rescalings = (1 + L) / (gamma_0**2 * theory_widths)
            theory_label = r'Theory ($(1+L)/\gamma_0^2 N$)'
        else:
            # Theory line for use_skips=False: L/(gamma_0^2 * N)
            theory_rescalings = L / (gamma_0**2 * theory_widths)
            theory_label = r'Theory ($L/\gamma_0^2 N$)'
        
        plt.plot(theory_widths, theory_rescalings, '--', color='black', linewidth=LINE_WIDTH,
                label=theory_label, alpha=0.8, zorder=2)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if output_dim > 1:
        plt.ylabel(r"$\|\mathbf{S}(\boldsymbol{\theta}_t)\|_2 - 1$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    else:
        plt.ylabel(r"$s(\boldsymbol{\theta}_t) - 1$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.legend(fontsize=FONT_SIZES["legend"])
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    save_plot(plot_dir, "rescaling_vs_width.pdf", None)


def plot_losses_and_energies(data, results_dir, colormap_name='viridis', n_hidden=None, log_x_scale=False, param_type=None, use_skips=None):
    """Plot DMFT loss, BP loss at highest width, and PC theory energy at different widths."""
    plt.figure(figsize=(12.5, 6))
    max_width = max(data["widths"])
    
    # Middle blue color for PC legend entry
    pc_legend_color = '#4A90E2'
    bp_color = '#DC143C'
    
    # Plot PC energies with Blues colormap (no labels)
    widths_list = sorted([w for w in data["widths"] if w in data["pc_energies"]])
    if widths_list:
        blues_cmap = plt.get_cmap('Blues')
        n_widths = len(widths_list)
        for idx, width in enumerate(widths_list):
            energies = np.array(data["pc_energies"][width]).flatten()
            iterations = np.arange(1, len(energies) + 1)
            color = blues_cmap(get_color_val(idx, n_widths, 'Blues'))
            plt.plot(iterations, energies, '-', alpha=ALPHA, linewidth=LINE_WIDTH, 
                    color=color, label='')
    
    # Plot BP loss at highest width (no label, single color)
    if max_width in data["bp_losses"]:
        bp_loss = np.array(data["bp_losses"][max_width]).flatten()
        iterations = np.arange(1, len(bp_loss) + 1)
        plt.plot(iterations, bp_loss, '-', color=bp_color, linewidth=LINE_WIDTH,
                alpha=ALPHA, label='')
    
    # Plot DMFT loss (skip for 'sp' param_type and when use_skips=True)
    if data["dmft_loss"] is not None and param_type != 'sp' and use_skips != True:
        dmft_loss = np.array(data["dmft_loss"]).flatten()
        iterations = np.arange(1, len(dmft_loss) + 1)
        plt.plot(iterations, dmft_loss, '--', color='black', linewidth=4,
                label=r'Theory ($N \rightarrow \infty$)', alpha=0.8)
    
    # Create custom legend
    legend_handles = []
    legend_labels = []
    
    # Add PC entry
    if widths_list:
        legend_handles.append(plt.Line2D([0], [0], color=pc_legend_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'$\mathcal{F}^*(\boldsymbol{\theta})$ (PC)')
    
    # Add BP entry
    if max_width in data["bp_losses"]:
        legend_handles.append(plt.Line2D([0], [0], color=bp_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'$\mathcal{L}(\boldsymbol{\theta})$ (BP)')
    
    # Add N entries with grayscale
    gray_cmap = plt.get_cmap('Greys')
    all_widths = sorted(set([w for w in widths_list] + ([max_width] if max_width in data["bp_losses"] else [])))
    n_all_widths = len(all_widths)
    for idx, width in enumerate(all_widths):
        gray_val = 0.3 + (idx / max(n_all_widths - 1, 1)) * 0.5 if n_all_widths > 1 else 0.5
        gray_color = gray_cmap(gray_val)
        legend_handles.append(plt.Line2D([0], [0], color=gray_color, linewidth=LINE_WIDTH))
        legend_labels.append(f'$N = {width}$')
    
    # Add DMFT entry if present (skip for 'sp' param_type and when use_skips=True)
    if data["dmft_loss"] is not None and param_type != 'sp' and use_skips != True:
        legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', alpha=0.8))
        legend_labels.append(r'Theory ($N \rightarrow \infty$)')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_x_scale:
        plt.xscale('log', base=10)
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    save_plot(results_dir, "losses_and_energies.pdf", n_hidden)


def plot_loss_energy_ratio(data, results_dir, colormap_name='viridis', loss_type='pc', n_hidden=None):
    """Plot ratio of loss to PC equilib energy over training for different widths."""
    if loss_type == 'pc':
        widths_list = sorted([w for w in data["widths"] 
                             if w in data["pc_energies"] and w in data["pc_train_losses"]])
        ylabel = r"$\mathcal{L}(\boldsymbol{\theta}_t) / \mathcal{F}^*(\boldsymbol{\theta}_t)$"
        filename = "pc_loss_energy_ratio.pdf"
    elif loss_type == 'bp':
        widths_list = sorted([w for w in data["widths"] 
                             if w in data["pc_energies"] and w in data["bp_losses"]])
        ylabel = r"$\mathcal{L}(\boldsymbol{\theta}_t) / \mathcal{F}^*(\boldsymbol{\theta}_t)$"
        filename = "bp_loss_energy_ratio.pdf"
    else:
        raise ValueError(f"loss_type must be 'pc' or 'bp', got '{loss_type}'")
    
    plt.figure(figsize=FIG_SIZE)
    if widths_list:
        colormap = plt.get_cmap(colormap_name)
        n_widths = len(widths_list)
        for idx, width in enumerate(widths_list):
            energies = np.array(data["pc_energies"][width]).flatten()
            losses = np.array(data["pc_train_losses" if loss_type == 'pc' else "bp_losses"][width]).flatten()
            n_iterations = min(len(energies), len(losses))
            ratio = losses[:n_iterations] / energies[:n_iterations]
            iterations = np.arange(1, len(ratio) + 1)
            color = colormap(get_color_val(idx, n_widths, colormap_name))
            plt.plot(iterations, ratio, label=f'$N = {width}$',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    setup_plot("$t$", ylabel)
    save_plot(results_dir, filename, n_hidden)


def plot_losses(data, results_dir, colormap_name='viridis', n_hidden=None, log_x_scale=False, param_type=None, use_skips=None):
    """Plot DMFT loss, BP loss at highest width, and PC train loss at different widths."""
    plt.figure(figsize=(12.5, 6))
    max_width = max(data["widths"])
    
    # Middle blue color for PC legend entry
    pc_legend_color = '#4A90E2'
    bp_color = '#DC143C'
    
    # Plot PC train losses with Blues colormap (no labels)
    widths_list = sorted([w for w in data["widths"] if w in data["pc_train_losses"]])
    if widths_list:
        blues_cmap = plt.get_cmap('Blues')
        n_widths = len(widths_list)
        for idx, width in enumerate(widths_list):
            train_losses = np.array(data["pc_train_losses"][width]).flatten()
            iterations = np.arange(1, len(train_losses) + 1)
            color = blues_cmap(get_color_val(idx, n_widths, 'Blues'))
            plt.plot(iterations, train_losses, '-', alpha=ALPHA, linewidth=LINE_WIDTH,
                    color=color, label='')
    
    # Plot BP and DMFT losses
    if max_width in data["bp_losses"]:
        bp_loss = np.array(data["bp_losses"][max_width]).flatten()
        iterations = np.arange(1, len(bp_loss) + 1)
        plt.plot(iterations, bp_loss, '-', color=bp_color, linewidth=LINE_WIDTH,
                alpha=ALPHA, label='')
    
    # Plot DMFT loss (skip for 'sp' param_type and when use_skips=True)
    if data["dmft_loss"] is not None and param_type != 'sp' and use_skips != True:
        dmft_loss = np.array(data["dmft_loss"]).flatten()
        iterations = np.arange(1, len(dmft_loss) + 1)
        plt.plot(iterations, dmft_loss, '--', color='black', linewidth=4,
                label=r'Theory ($N \rightarrow \infty$)', alpha=0.8)
    
    # Create custom legend
    legend_handles = []
    legend_labels = []
    
    # Add PC entry
    if widths_list:
        legend_handles.append(plt.Line2D([0], [0], color=pc_legend_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'PC')
    
    # Add BP entry
    if max_width in data["bp_losses"]:
        legend_handles.append(plt.Line2D([0], [0], color=bp_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'BP')
    
    # Add N entries with grayscale
    gray_cmap = plt.get_cmap('Greys')
    all_widths = sorted(set([w for w in widths_list] + ([max_width] if max_width in data["bp_losses"] else [])))
    n_all_widths = len(all_widths)
    for idx, width in enumerate(all_widths):
        gray_val = 0.3 + (idx / max(n_all_widths - 1, 1)) * 0.5 if n_all_widths > 1 else 0.5
        gray_color = gray_cmap(gray_val)
        legend_handles.append(plt.Line2D([0], [0], color=gray_color, linewidth=LINE_WIDTH))
        legend_labels.append(f'$N = {width}$')
    
    # Add DMFT entry if present (skip for 'sp' param_type and when use_skips=True)
    if data["dmft_loss"] is not None and param_type != 'sp' and use_skips != True:
        legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', alpha=0.8))
        legend_labels.append(r'Theory ($N \rightarrow \infty$)')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_x_scale:
        plt.xscale('log', base=10)
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    plt.yscale('log', base=10)
    save_plot(results_dir, "losses.pdf", n_hidden)


def plot_bp_pc_losses_comparison(data, plot_dir, colormap_name='viridis', n_hidden=None, narrow_width=None,
                                  seed=0, data_results_dir=None, param_type=None, use_skips=None):
    """Plot BP loss and PC loss for smallest and largest width over training.
    
    Args:
        narrow_width: Specific width to use for the narrow network. If None, uses the minimum available width.
        data_results_dir: Directory to discover all available widths from. If None, uses only widths from data.
        param_type: Parameter type to filter by ('sp' or 'mupc', None to load all)
    """
    plt.figure(figsize=FIG_SIZE)
    
    # Discover all available widths if data_results_dir is provided
    if data_results_dir is not None:
        print("  Discovering all available widths for bp_pc_losses_comparison plot...")
        all_available_widths = discover_available_widths(data_results_dir, seed=seed, n_hidden=n_hidden,
                                                           required_files=["train_losses.npy", "losses.npy"],
                                                           param_type=param_type, use_skips=use_skips)
        if all_available_widths:
            print(f"  Found {len(all_available_widths)} widths with loss data: {all_available_widths}")
            all_widths = all_available_widths
        else:
            print("  Warning: No widths discovered, falling back to data widths")
            all_widths = sorted(set([w for w in data["widths"] 
                                     if w in data["pc_train_losses"] or w in data["bp_losses"]]))
    else:
        all_widths = sorted(set([w for w in data["widths"] 
                                 if w in data["pc_train_losses"] or w in data["bp_losses"]]))
    
    if not all_widths:
        print("  Warning: No data available for BP/PC losses comparison")
        plt.close()
        return
    
    # Determine narrow and wide widths
    if narrow_width is not None:
        if narrow_width not in all_widths:
            print(f"  Warning: Specified narrow width {narrow_width} not available. Available widths: {all_widths}")
            print(f"  Using minimum available width instead: {min(all_widths)}")
            min_width = min(all_widths)
        else:
            min_width = narrow_width
    else:
        min_width = min(all_widths)
    
    max_width = max(all_widths)
    
    # Load loss data for selected widths if not already in data
    losses_data = {"pc_train_losses": data["pc_train_losses"].copy(), "bp_losses": data["bp_losses"].copy()}
    if data_results_dir is not None:
        for width in [min_width, max_width]:
            if width not in losses_data["pc_train_losses"] or width not in losses_data["bp_losses"]:
                seed_str = str(seed)
                n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
                param_type_str = f"{param_type}_param_type" if param_type else None
                use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
                width_str = f"{width}_width"
                
                for root, _, files in os.walk(data_results_dir):
                    if width_str in root and root.split(os.sep)[-1] == seed_str:
                        if n_hidden_str is None or n_hidden_str in root:
                            # Filter by param_type if specified
                            if param_type_str is None or param_type_str in root:
                                # Filter by use_skips if specified
                                if use_skips_str is None or use_skips_str in root:
                                    # Load PC train losses
                                    if "train_losses.npy" in files and width not in losses_data["pc_train_losses"]:
                                        arr = _load_npy_safe(os.path.join(root, "train_losses.npy"))
                                        if arr is not None:
                                            losses_data["pc_train_losses"][width] = arr
                                    # Load BP losses
                                    if "losses.npy" in files and width not in losses_data["bp_losses"]:
                                        arr = _load_npy_safe(os.path.join(root, "losses.npy"))
                                        if arr is not None:
                                            losses_data["bp_losses"][width] = arr
    colors = {'pc_min': '#1E88E5', 'pc_max': '#90CAF9', 'bp_min': '#8B0000', 'bp_max': '#EF5350'}
    
    # Plot in order: PC small, BP small, PC large, BP large
    # Inverted colors: narrow networks use max colors, wide networks use min colors
    for width, prefix in [(min_width, 'min'), (max_width, 'max')]:
        # Invert: min_width uses max colors, max_width uses min colors
        color_prefix = 'max' if prefix == 'min' else 'min'
        
        # Plot PC first, then BP for each width
        if width in losses_data["pc_train_losses"]:
            pc_loss = np.array(losses_data["pc_train_losses"][width]).flatten()
            iterations = np.arange(1, len(pc_loss) + 1)
            plt.plot(iterations, pc_loss, '-', color=colors[f'pc_{color_prefix}'],
                    linewidth=4.5, label=f'PC, $N = {width}$', alpha=0.85)
        
        if width in losses_data["bp_losses"]:
            bp_loss = np.array(losses_data["bp_losses"][width]).flatten()
            iterations = np.arange(1, len(bp_loss) + 1)
            plt.plot(iterations, bp_loss, '--', color=colors[f'bp_{color_prefix}'],
                    linewidth=4.5, label=f'BP, $N = {width}$', alpha=0.85)
    
    setup_plot("$t$", r"$\mathcal{L}(\boldsymbol{\theta}_t)$", log_scale=True)
    save_plot(plot_dir, "bp_pc_losses_comparison.pdf", n_hidden, add_suffix=False)


def calculate_gradient_metrics(data):
    """Calculate rescaled loss gradient and extra energy gradient term."""
    metrics = {
        "rescaled_loss_grad_norms": {},
        "extra_energy_grad_norms": {}
    }
    
    for width in data["widths"]:
        if width not in data["pc_grads"] or width not in data["bp_grads"]:
            continue
        if width not in data["pc_rescalings"]:
            continue
        
        pc_grads = np.array(data["pc_grads"][width])  # Shape: (n_iterations, n_params)
        bp_grads = np.array(data["bp_grads"][width])  # Shape: (n_iterations, n_params)
        rescalings = np.array(data["pc_rescalings"][width])  # Shape: (n_iterations,) - scalar per iteration
        
        # Ensure rescalings is 1D array of scalars
        if rescalings.ndim == 0:
            rescalings = np.array([rescalings])
        rescalings = rescalings.flatten()
        
        # Ensure same number of iterations
        n_iterations = min(len(pc_grads), len(bp_grads), len(rescalings))
        pc_grads = pc_grads[:n_iterations]
        bp_grads = bp_grads[:n_iterations]
        rescalings = rescalings[:n_iterations]
        
        # Calculate rescaled loss gradient: BP grads / rescaling
        # rescalings is (n_iterations,), bp_grads is (n_iterations, n_params)
        # Broadcasting: divide each row of bp_grads by corresponding rescaling value
        rescaled_loss_grads = bp_grads / rescalings[:, np.newaxis]  # Shape: (n_iterations, n_params)
        
        # Calculate extra energy gradient term: PC grad - rescaled loss grad
        extra_energy_grads = pc_grads - rescaled_loss_grads
        
        # Calculate norms
        rescaled_loss_grad_norms = np.linalg.norm(rescaled_loss_grads, axis=1)
        extra_energy_grad_norms = np.linalg.norm(extra_energy_grads, axis=1)
        
        metrics["rescaled_loss_grad_norms"][width] = rescaled_loss_grad_norms
        metrics["extra_energy_grad_norms"][width] = extra_energy_grad_norms
    
    return metrics


def plot_gradient_norms(data, metrics, results_dir, colormap_name='viridis', n_hidden=None):
    """Plot norms of rescaled loss gradient and extra energy gradient term."""
    plots = [
        ("rescaled_loss_grad_norms", r"$\left\|\frac{1}{s(\boldsymbol{\theta})}\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\right\|$", "rescaled_loss_grad_norms.pdf"),
        ("extra_energy_grad_norms", r"$\left\|\frac{1}{s(\boldsymbol{\theta})^2} \mathcal{L} \frac{\partial s}{\partial \boldsymbol{\theta}}\right\|$", "extra_energy_grad_norms.pdf")
    ]
    
    for metric_key, ylabel, filename in plots:
        plt.figure(figsize=FIG_SIZE)
        widths_list = sorted([w for w in data["widths"] if w in metrics[metric_key]])
        if widths_list:
            colormap = plt.get_cmap(colormap_name)
            n_widths = len(widths_list)
            for idx, width in enumerate(widths_list):
                norms = metrics[metric_key][width]
                iterations = np.arange(1, len(norms) + 1)
                color = colormap(get_color_val(idx, n_widths, colormap_name))
                plt.plot(iterations, norms, label=f'$N = {width}$',
                        alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        setup_plot("$t$", ylabel, log_scale=True)
        save_plot(results_dir, filename, n_hidden)


def calculate_cosine_similarity(data):
    """Extract pre-computed cosine similarities from loaded data.
    
    Cosine similarities are computed and saved during training, so we just
    extract them from the loaded data dictionary.
    """
    similarities = {}
    for width in data["widths"]:
        if width in data["grad_cosine_similarities"]:
            similarities[width] = data["grad_cosine_similarities"][width]
    return similarities


def plot_cosine_similarity(data, similarities, results_dir, colormap_name='viridis', n_hidden=None):
    """Plot cosine similarity between BP grads and PC grads."""
    plt.figure(figsize=FIG_SIZE)
    widths_list = sorted([w for w in data["widths"] if w in similarities])
    
    if widths_list:
        colormap = plt.get_cmap(colormap_name)
        n_widths = len(widths_list)
        
        for idx, width in enumerate(widths_list):
            values = np.array(similarities[width]).flatten()
            iterations = np.arange(1, len(values) + 1)
            color = colormap(get_color_val(idx, n_widths, colormap_name))
            plt.plot(iterations, values, label=f'$N = {width}$',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}^*\right)$", 
               fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, labels=labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.0, 0.0), loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    save_plot(results_dir, "grads_cosine_similarities.pdf", n_hidden)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling limits results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory containing experiment outputs"
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs='+',
        default=[32, 128, 512, 2048],  
        help="List of widths N to plot"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed value for loading data"
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["all", "rescalings", "rescaling_vs_width", "losses", "energies", "ratio", "grads", "cosine", "comparison"],
        default="all",
        help="Which plot(s) to generate"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="Blues",
        help="Colormap to use for plotting (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'tab10')"
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        nargs='+',
        default=[4],
        help="List of hidden layer counts H to plot. If not provided, plots without n_hidden filtering."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="toy_plots",
        help="Directory to save plots. Will create subdirectories for dataset ID and then for each n_hidden."
    )
    parser.add_argument(
        "--gamma_0",
        type=float,
        default=None,
        help="Gamma_0 value for theoretical line (default: extracted from data directory)"
    )
    parser.add_argument(
        "--narrow_width_comparison",
        type=int,
        default=32,
        help="Width to use for the narrow network in BP/PC losses comparison plot (default: use minimum available width)"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=1,
        help="Output dimension. If > 1, labels will use spectral norm notation (default: 1)"
    )
    parser.add_argument(
        "--log_x_scale",
        action="store_true",
        default=False,
        help="Use log base 10 scale for x-axis in losses and losses_and_energies plots"
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
        help="Filter results by use_skips setting. Use --use_skips True/False or --use_skips true/false. If not provided, plots both True and False."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=["toy"],
        help="List of dataset names to plot. Plots will be saved in plot_dir/dataset/. (default: ['toy'])"
    )
    parser.add_argument(
        "--dataset_input_dims",
        type=int,
        nargs='+',
        default=[40],
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
    
    # Set n_hiddens to process
    if args.n_hiddens is None:
        n_hiddens = [None]
        print("No n_hiddens specified. Proceeding without n_hidden filtering.")
    else:
        n_hiddens = args.n_hiddens
        print(f"Processing plots for n_hiddens: {n_hiddens}")
    
    # Create base plot directory
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Define param types to process
    param_types = ['sp', 'mupc' , 'my-mup']
    
    # Define use_skips values to process
    # If args.use_skips is specified, only process that value; otherwise process both True and False
    if args.use_skips is not None:
        use_skips_values = [args.use_skips]
    else:
        use_skips_values = [True, False]
    
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
        
        # Loop over each param type
        for param_type in param_types:
            print(f"\n{'='*70}")
            print(f"Processing plots for param_type = {param_type}")
            print(f"{'='*70}\n")
            
            # Loop over each n_hidden configuration
            for n_hidden in n_hiddens:
                print(f"\n{'='*60}")
                if n_hidden is not None:
                    print(f"Processing plots for H = {n_hidden}, param_type = {param_type}")
                else:
                    print(f"Processing plots (no n_hidden filter), param_type = {param_type}")
                print(f"{'='*60}\n")
                
                # Create subdirectory for this n_hidden within the dataset directory
                if n_hidden is not None:
                    n_hidden_dir = os.path.join(dataset_plot_dir, f"{n_hidden}_n_hidden")
                else:
                    n_hidden_dir = os.path.join(dataset_plot_dir, "all")
                os.makedirs(n_hidden_dir, exist_ok=True)
                
                # Loop over each use_skips configuration
                for use_skips in use_skips_values:
                    # Create subdirectory for this skip status
                    skip_status_dir = os.path.join(n_hidden_dir, f"{use_skips}_use_skips")
                    os.makedirs(skip_status_dir, exist_ok=True)
                    
                    # Create subdirectory for this param_type
                    plot_dir = os.path.join(skip_status_dir, param_type)
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    print(f"\n{'='*50}")
                    print(f"Processing plots for use_skips = {use_skips}, param_type = {param_type}")
                    if n_hidden is not None:
                        print(f"H = {n_hidden}")
                    print(f"{'='*50}\n")
                    
                    # Load data for this n_hidden, param_type, and use_skips
                    print(f"Loading data from {dataset_results_dir} (param_type={param_type}, use_skips={use_skips})...")
                    data = load_data_from_dir(dataset_results_dir, args.widths, seed=args.seed, n_hidden=n_hidden, param_type=param_type, use_skips=use_skips)
                    
                    # Print summary of loaded data
                    print(f"\nLoaded data summary:")
                    print(f"  DMFT loss: {'Yes' if data['dmft_loss'] is not None else 'No'}")
                    print(f"  PC rescalings: {sorted(data['pc_rescalings'].keys())}")
                    print(f"  PC energies: {sorted(data['pc_energies'].keys())}")
                    print(f"  PC train losses: {sorted(data['pc_train_losses'].keys())}")
                    print(f"  PC grads: {sorted(data['pc_grads'].keys())}")
                    print(f"  BP losses: {sorted(data['bp_losses'].keys())}")
                    print(f"  BP grads: {sorted(data['bp_grads'].keys())}")
                    print(f"  Cosine similarities: {sorted(data['grad_cosine_similarities'].keys())}")
                    print()
                    
                    # Generate plots
                    plot_configs = [
                        ("rescalings", plot_rescalings, [data, plot_dir, args.colormap, None, args.output_dim], "pc_rescaling.pdf"),
                        ("rescaling_vs_width", plot_rescaling_vs_width, [data, plot_dir, args.colormap, n_hidden, args.gamma_0, args.seed, True, dataset_results_dir, args.output_dim, param_type, use_skips], "rescaling_vs_width.pdf"),
                        ("energies", plot_losses_and_energies, [data, plot_dir, args.colormap, None, args.log_x_scale, param_type, use_skips], "losses_and_energies.pdf"),
                        ("losses", plot_losses, [data, plot_dir, args.colormap, None, args.log_x_scale, param_type, use_skips], "losses.pdf"),
                        ("comparison", plot_bp_pc_losses_comparison, [data, plot_dir, args.colormap, n_hidden, args.narrow_width_comparison, args.seed, dataset_results_dir, param_type, use_skips], "bp_pc_losses_comparison.pdf"),
                    ]
                    
                    for plot_type, plot_func, plot_args, filename in plot_configs:
                        if args.plot == "all" or args.plot == plot_type:
                            print(f"Generating {plot_type} plot...")
                            plot_func(*plot_args)
                            if plot_type == "rescaling_vs_width" and data.get("gamma_0") is not None:
                                print(f"  Using gamma_0 = {data['gamma_0']} from data directory")
                            print(f"  Saved to {os.path.join(plot_dir, filename)}")
                    
                    if args.plot == "all" or args.plot == "ratio":
                        for loss_type, filename in [("pc", "pc_loss_energy_ratio.pdf"), ("bp", "bp_loss_energy_ratio.pdf")]:
                            print(f"Generating {loss_type} loss/energy ratio plot...")
                            plot_loss_energy_ratio(data, plot_dir, args.colormap, loss_type, None)
                            print(f"  Saved to {os.path.join(plot_dir, filename)}")
                    
                    if args.plot == "all" or args.plot == "cosine":
                        print("Extracting cosine similarities...")
                        similarities = calculate_cosine_similarity(data)
                        if similarities:
                            print("Generating cosine similarity plot...")
                            plot_cosine_similarity(data, similarities, plot_dir, args.colormap, None)
                            print(f"  Saved to {os.path.join(plot_dir, 'grads_cosine_similarities.pdf')}")
                        else:
                            print("  Warning: No cosine similarity data found")
            
            print(f"\nCompleted processing for param_type = {param_type}")
        
        print(f"\nCompleted processing for dataset: {dataset}")
    
    print("\nDone!")
