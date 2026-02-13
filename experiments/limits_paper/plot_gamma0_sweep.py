import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def get_color_val(idx, n_gammas, colormap_name):
    """Get color value for colormap."""
    if is_sequential_colormap(colormap_name):
        return 0.15 + (idx / max(n_gammas - 1, 1)) * 0.85 if n_gammas > 1 else 0.15
    return (idx / max(n_gammas - 1, 1)) * 0.85 if n_gammas > 1 else 0


def setup_plot(xlabel, ylabel, log_scale=False):
    """Setup common plot styling."""
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
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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


def plot_gammas_series(data, data_key, results_dir, colormap_name, ylabel, filename, 
                       n_hidden=None, log_scale=False, label_prefix=r'$\gamma_0 = {}$'):
    """Generic function to plot a series over gamma_0 values."""
    plt.figure(figsize=FIG_SIZE)
    gammas_list = sorted([g for g in data["gamma_0s"] if g in data[data_key]])
    
    if gammas_list:
        colormap = plt.get_cmap(colormap_name)
        n_gammas = len(gammas_list)
        
        for idx, gamma_0 in enumerate(gammas_list):
            values = np.array(data[data_key][gamma_0]).flatten()
            iterations = np.arange(1, len(values) + 1)
            color = colormap(get_color_val(idx, n_gammas, colormap_name))
            plt.plot(iterations, values, label=label_prefix.format(gamma_0),
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    setup_plot("$t$", ylabel, log_scale)
    save_plot(results_dir, filename, n_hidden, add_suffix=False)


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


def discover_available_gamma_0s(results_dir, width, seed=0, n_hidden=None, param_type=None, required_files=None, use_skips=None):
    """Discover all available gamma_0 values from the results directory structure.
    
    Args:
        results_dir: Directory to search
        width: Width value to match
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        param_type: Parameter type to filter by ('sp', 'mupc', 'ntp', None for any)
        required_files: List of file names that must be present. If None, defaults to ["loss_rescalings.npy"]
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    """
    if required_files is None:
        required_files = ["loss_rescalings.npy"]
    
    gamma_0s = set()
    seed_str = str(seed)
    width_str = f"{width}_width"
    n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    for root, _, files in os.walk(results_dir):
        # Check if any of the required files are present
        if any(req_file in files for req_file in required_files):
            # Check if this directory matches our width, seed, n_hidden, param_type, and use_skips criteria
            if width_str in root and root.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in root:
                    if param_type_str is None or param_type_str in root:
                        if use_skips_str is None or use_skips_str in root:
                            # Extract gamma_0 from directory path
                            for part in root.split(os.sep):
                                if part.endswith("_gamma_0"):
                                    try:
                                        gamma_0 = float(part.replace("_gamma_0", ""))
                                        gamma_0s.add(gamma_0)
                                        break
                                    except ValueError:
                                        pass
    
    return sorted(list(gamma_0s))


def load_data_from_dir(results_dir, gamma_0s, width, seed=0, n_hidden=None, param_type=None, use_skips=None):
    """Load all data from the results directory structure for a given width and varying gamma_0.
    
    Args:
        results_dir: Directory to search
        gamma_0s: List of gamma_0 values to load
        width: Width value to match
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        param_type: Parameter type to filter by ('sp', 'mupc', 'ntp', None for any)
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    """
    data = {
        "gamma_0s": gamma_0s, "width": width, "pc_rescalings": {}, "pc_energies": {},
        "pc_train_losses": {}, "pc_grads": {}, "bp_losses": {},
        "bp_grads": {}, "grad_cosine_similarities": {}, "dmft_loss": None
    }
    
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    
    # Check for Adam optimizer (filter by param_type if specified)
    use_adam = any(
        ("adam_param_optim" in root or "adam_optim_id" in root) and 
        (param_type_str is None or param_type_str in root)
        for root, _, _ in os.walk(results_dir)
    )
    
    # Load DMFT loss - try different naming patterns
    # Store DMFT loss per gamma_0 (they should all be the same, but load one for each)
    data["dmft_losses"] = {}
    if not use_adam:
        # Extract dataset ID from results directory
        dataset_id = extract_dataset_id(results_dir)
        
        # Try loading DMFT loss for each gamma_0
        print(f"  Attempting to load DMFT loss for gamma_0s: {gamma_0s}")
        for gamma_0 in gamma_0s:
                # Try patterns with gamma_0
                loaded = False
                for path in [f"dmft_loss_{gamma_0}_gamma_0_seed_{seed}.npy",
                             f"dmft_loss_{dataset_id}_{gamma_0}_gamma_0_seed_{seed}.npy",
                             f"dmft_loss_{dataset_id}_n_hidden_{n_hidden}_{gamma_0}_gamma_0_seed_{seed}.npy"]:
                    full_path = os.path.join(results_dir, path)
                    dmft = _load_npy_safe(full_path)
                    if dmft is not None:
                        data["dmft_losses"][gamma_0] = dmft
                        if data["dmft_loss"] is None:
                            data["dmft_loss"] = dmft
                        print(f"    Loaded DMFT for gamma_0={gamma_0} from: {path}")
                        loaded = True
                        break
                if not loaded:
                    print(f"    Could not load DMFT for gamma_0={gamma_0}")
        
        # If not found with gamma_0, try old patterns without gamma_0
        if not data["dmft_losses"] and n_hidden is not None:
            print(f"  Trying old DMFT patterns without gamma_0...")
            for path in [f"dmft_loss_{dataset_id}_n_hidden_{n_hidden}_seed_{seed}.npy",
                         f"dmft_loss_n_hidden_{n_hidden}_seed_{seed}.npy",
                         f"dmft_loss_seed_{seed}.npy"]:
                full_path = os.path.join(results_dir, path)
                dmft = _load_npy_safe(full_path)
                if dmft is not None:
                    data["dmft_loss"] = dmft
                    # Use the same DMFT loss for all gamma_0s if found without gamma_0 in filename
                    for gamma_0 in gamma_0s:
                        data["dmft_losses"][gamma_0] = dmft
                    print(f"    Loaded DMFT from: {path} (using for all gamma_0s)")
                    break
    
    # Build directory map
    dir_map = {}
    for root, _, files in os.walk(results_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            dir_map[root] = npy_files
    
    # Load data for each gamma_0
    width_str, seed_str = f"{width}_width", str(seed)
    n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
    
    for gamma_0 in gamma_0s:
        gamma_0_str = f"{gamma_0}_gamma_0"
        
        pc_dir = bp_dir = None
        for dir_path, files in dir_map.items():
            if width_str in dir_path and gamma_0_str in dir_path and dir_path.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in dir_path:
                    if param_type_str is None or param_type_str in dir_path:
                        if use_skips_str is None or use_skips_str in dir_path:
                            if "energies.npy" in files:
                                pc_dir = dir_path
                            elif "losses.npy" in files:
                                bp_dir = dir_path
        
        # Load PC data
        if pc_dir:
            pc_files = {
                "pc_energies": "energies.npy",
                "pc_train_losses": "train_losses.npy",
                "pc_rescalings": "loss_rescalings.npy"
            }
            for key, filename in pc_files.items():
                arr = _load_npy_safe(os.path.join(pc_dir, filename))
                if arr is not None:
                    data[key][gamma_0] = arr
            
            grads = _load_npy_safe(os.path.join(pc_dir, "grads.npy"), flatten=False)
            if grads is not None:
                data["pc_grads"][gamma_0] = grads
            
            # Load cosine similarities if available (preferred over computing from gradients)
            cosine_sims = _load_npy_safe(os.path.join(pc_dir, "grad_cosine_similarities.npy"))
            if cosine_sims is not None:
                data["grad_cosine_similarities"][gamma_0] = cosine_sims
        
        # Load BP data
        if bp_dir:
            for key, filename in [("bp_losses", "losses.npy"), ("bp_grads", "bp_grads.npy")]:
                arr = _load_npy_safe(os.path.join(bp_dir, filename), flatten=(key == "bp_losses"))
                if arr is not None:
                    data[key][gamma_0] = arr
    
    return data


def plot_rescalings(data, results_dir, colormap_name='viridis', n_hidden=None, output_dim=1):
    """Plot PC rescalings S during training as a function of different gamma_0 values."""
    if output_dim > 1:
        ylabel = r"$\|\mathbf{S}(\boldsymbol{\theta}_t)\|_2$"
    else:
        ylabel = r"$s(\boldsymbol{\theta}_t)$"
    plot_gammas_series(data, "pc_rescalings", results_dir, colormap_name,
                       ylabel, "pc_rescaling.pdf", n_hidden)


def plot_losses_and_energies(data, results_dir, colormap_name='viridis', n_hidden=None, log_x_scale=False, plot_theory=False):
    """Plot DMFT loss, BP loss, and PC theory energy at different gamma_0 values."""
    plt.figure(figsize=(10, 6))
    
    # Get all available gamma_0s for both PC and BP
    pc_gammas_list = sorted([g for g in data["gamma_0s"] if g in data["pc_energies"]])
    bp_gammas_list = sorted([g for g in data["gamma_0s"] if g in data["bp_losses"]])
    all_gammas = sorted(set(pc_gammas_list + bp_gammas_list))
    n_gammas = len(all_gammas) if all_gammas else 0
    
    if all_gammas:
        # Plot PC energies at different gamma_0s (blue shades)
        blue_colormap = plt.get_cmap('Blues')
        for gamma_0 in pc_gammas_list:
            energies = np.array(data["pc_energies"][gamma_0]).flatten()
            iterations = np.arange(1, len(energies) + 1)
            idx = all_gammas.index(gamma_0)
            color_val = get_color_val(idx, n_gammas, 'Blues')
            color = blue_colormap(color_val)
            plt.plot(iterations, energies, '-', alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        
        # Plot BP loss at all gamma_0s (red shades)
        red_colormap = plt.get_cmap('Reds')
        for gamma_0 in bp_gammas_list:
            bp_loss = np.array(data["bp_losses"][gamma_0]).flatten()
            iterations = np.arange(1, len(bp_loss) + 1)
            idx = all_gammas.index(gamma_0)
            color_val = get_color_val(idx, n_gammas, 'Reds')
            color = red_colormap(color_val)
            plt.plot(iterations, bp_loss, '--', alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        
        # Plot DMFT loss for each gamma_0 (same value, but plot for each with greyscale)
        if plot_theory:
            # Use all gamma_0s from data, not just those with PC/BP data
            all_gammas_for_dmft = sorted(data["gamma_0s"])
            n_gammas_for_dmft = len(all_gammas_for_dmft) if all_gammas_for_dmft else 0
            if "dmft_losses" in data and data["dmft_losses"]:
                for idx, gamma_0 in enumerate(all_gammas_for_dmft):
                    if gamma_0 in data["dmft_losses"]:
                        dmft_loss = np.array(data["dmft_losses"][gamma_0]).flatten()
                        dmft_iterations = np.arange(1, len(dmft_loss) + 1)
                        grey_val = 0.8 - (idx / (n_gammas_for_dmft - 1)) * 0.6 if n_gammas_for_dmft > 1 else 0.5
                        grey_color = (grey_val, grey_val, grey_val)
                        plt.plot(dmft_iterations, dmft_loss, ':', color=grey_color, linewidth=4, alpha=0.8)
            elif data["dmft_loss"] is not None:
                # Fallback to single DMFT loss if per-gamma_0 loading didn't work
                dmft_loss = np.array(data["dmft_loss"]).flatten()
                dmft_iterations = np.arange(1, len(dmft_loss) + 1)
                for idx, gamma_0 in enumerate(all_gammas_for_dmft):
                    grey_val = 0.8 - (idx / (n_gammas_for_dmft - 1)) * 0.6 if n_gammas_for_dmft > 1 else 0.5
                    grey_color = (grey_val, grey_val, grey_val)
                    plt.plot(dmft_iterations, dmft_loss, ':', color=grey_color, linewidth=4, alpha=0.8)
    
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$\mathcal{F}^*(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_x_scale:
        plt.xscale('log', base=10)
    
    # Create custom legend
    legend_elements = []
    blue_colormap = plt.get_cmap('Blues')
    pc_middle_color = blue_colormap(0.5)
    legend_elements.append(Line2D([0], [0], color=pc_middle_color, linestyle='-', linewidth=LINE_WIDTH, label='PC'))
    red_colormap = plt.get_cmap('Reds')
    bp_middle_color = red_colormap(0.5)
    legend_elements.append(Line2D([0], [0], color=bp_middle_color, linestyle='--', linewidth=LINE_WIDTH, label='BP'))
    if all_gammas:
        for idx, gamma_0 in enumerate(all_gammas):
            grey_val = 0.8 - (idx / (n_gammas - 1)) * 0.6 if n_gammas > 1 else 0.5
            grey_color = (grey_val, grey_val, grey_val)
            legend_elements.append(Line2D([0], [0], color=grey_color, linestyle='-', linewidth=LINE_WIDTH, label=f'$\gamma_0 = {gamma_0}$'))
    if plot_theory and (("dmft_losses" in data and data["dmft_losses"]) or data["dmft_loss"] is not None):
        legend_elements.append(Line2D([0], [0], color='black', linestyle=':', linewidth=4, label=r'Theory ($N \rightarrow \infty$)'))
    
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_plot(results_dir, "losses_and_energies.pdf", n_hidden, add_suffix=False)


def plot_loss_energy_ratio(data, results_dir, colormap_name='viridis', loss_type='pc', n_hidden=None):
    """Plot ratio of loss to PC equilib energy over training for different gamma_0 values."""
    if loss_type == 'pc':
        gammas_list = sorted([g for g in data["gamma_0s"] 
                             if g in data["pc_energies"] and g in data["pc_train_losses"]])
        ylabel = r"$\mathcal{L}(\boldsymbol{\theta}_t) / \mathcal{F}^*(\boldsymbol{\theta}_t)$"
        filename = "pc_loss_energy_ratio.pdf"
    elif loss_type == 'bp':
        gammas_list = sorted([g for g in data["gamma_0s"] 
                             if g in data["pc_energies"] and g in data["bp_losses"]])
        ylabel = r"$\mathcal{L}(\boldsymbol{\theta}_t) / \mathcal{F}^*(\boldsymbol{\theta}_t)$"
        filename = "bp_loss_energy_ratio.pdf"
    else:
        raise ValueError(f"loss_type must be 'pc' or 'bp', got '{loss_type}'")
    
    plt.figure(figsize=FIG_SIZE)
    if gammas_list:
        colormap = plt.get_cmap(colormap_name)
        n_gammas = len(gammas_list)
        for idx, gamma_0 in enumerate(gammas_list):
            energies = np.array(data["pc_energies"][gamma_0]).flatten()
            losses = np.array(data["pc_train_losses" if loss_type == 'pc' else "bp_losses"][gamma_0]).flatten()
            n_iterations = min(len(energies), len(losses))
            ratio = losses[:n_iterations] / energies[:n_iterations]
            iterations = np.arange(1, len(ratio) + 1)
            color = colormap(get_color_val(idx, n_gammas, colormap_name))
            plt.plot(iterations, ratio, label=f'$\gamma_0 = {gamma_0}$',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    setup_plot("$t$", ylabel)
    save_plot(results_dir, filename, n_hidden, add_suffix=False)


def plot_losses(data, results_dir, colormap_name='viridis', n_hidden=None, log_x_scale=False, plot_theory=False):
    """Plot DMFT loss, BP loss, and PC train loss at different gamma_0 values."""
    plt.figure(figsize=(10, 6))
    
    # Get all available gamma_0s for both PC and BP
    pc_gammas_list = sorted([g for g in data["gamma_0s"] if g in data["pc_train_losses"]])
    bp_gammas_list = sorted([g for g in data["gamma_0s"] if g in data["bp_losses"]])
    all_gammas = sorted(set(pc_gammas_list + bp_gammas_list))
    n_gammas = len(all_gammas) if all_gammas else 0
    
    if all_gammas:
        # Plot PC train losses at different gamma_0s (blue shades)
        blue_colormap = plt.get_cmap('Blues')
        for gamma_0 in pc_gammas_list:
            train_losses = np.array(data["pc_train_losses"][gamma_0]).flatten()
            iterations = np.arange(1, len(train_losses) + 1)
            idx = all_gammas.index(gamma_0)
            color_val = get_color_val(idx, n_gammas, 'Blues')
            color = blue_colormap(color_val)
            plt.plot(iterations, train_losses, '-', alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        
        # Plot BP loss at all gamma_0s (red shades)
        red_colormap = plt.get_cmap('Reds')
        for gamma_0 in bp_gammas_list:
            bp_loss = np.array(data["bp_losses"][gamma_0]).flatten()
            iterations = np.arange(1, len(bp_loss) + 1)
            idx = all_gammas.index(gamma_0)
            color_val = get_color_val(idx, n_gammas, 'Reds')
            color = red_colormap(color_val)
            plt.plot(iterations, bp_loss, '--', alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        
        # Plot DMFT loss for each gamma_0 (same value, but plot for each with greyscale)
        if plot_theory:
            # Use all gamma_0s from data, not just those with PC/BP data
            all_gammas_for_dmft = sorted(data["gamma_0s"])
            n_gammas_for_dmft = len(all_gammas_for_dmft) if all_gammas_for_dmft else 0
            if "dmft_losses" in data and data["dmft_losses"]:
                for idx, gamma_0 in enumerate(all_gammas_for_dmft):
                    if gamma_0 in data["dmft_losses"]:
                        dmft_loss = np.array(data["dmft_losses"][gamma_0]).flatten()
                        dmft_iterations = np.arange(1, len(dmft_loss) + 1)
                        grey_val = 0.8 - (idx / (n_gammas_for_dmft - 1)) * 0.6 if n_gammas_for_dmft > 1 else 0.5
                        grey_color = (grey_val, grey_val, grey_val)
                        plt.plot(dmft_iterations, dmft_loss, ':', color=grey_color, linewidth=4, alpha=0.8)
            elif data["dmft_loss"] is not None:
                # Fallback to single DMFT loss if per-gamma_0 loading didn't work
                dmft_loss = np.array(data["dmft_loss"]).flatten()
                dmft_iterations = np.arange(1, len(dmft_loss) + 1)
                for idx, gamma_0 in enumerate(all_gammas_for_dmft):
                    grey_val = 0.8 - (idx / (n_gammas_for_dmft - 1)) * 0.6 if n_gammas_for_dmft > 1 else 0.5
                    grey_color = (grey_val, grey_val, grey_val)
                    plt.plot(dmft_iterations, dmft_loss, ':', color=grey_color, linewidth=4, alpha=0.8)
    
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_x_scale:
        plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    
    # Create custom legend
    legend_elements = []
    blue_colormap = plt.get_cmap('Blues')
    pc_middle_color = blue_colormap(0.5)
    legend_elements.append(Line2D([0], [0], color=pc_middle_color, linestyle='-', linewidth=LINE_WIDTH, label='PC'))
    red_colormap = plt.get_cmap('Reds')
    bp_middle_color = red_colormap(0.5)
    legend_elements.append(Line2D([0], [0], color=bp_middle_color, linestyle='--', linewidth=LINE_WIDTH, label='BP'))
    if all_gammas:
        for idx, gamma_0 in enumerate(all_gammas):
            grey_val = 0.8 - (idx / (n_gammas - 1)) * 0.6 if n_gammas > 1 else 0.5
            grey_color = (grey_val, grey_val, grey_val)
            legend_elements.append(Line2D([0], [0], color=grey_color, linestyle='-', linewidth=LINE_WIDTH, label=f'$\gamma_0 = {gamma_0}$'))
    if plot_theory and (("dmft_losses" in data and data["dmft_losses"]) or data["dmft_loss"] is not None):
        legend_elements.append(Line2D([0], [0], color='black', linestyle=':', linewidth=4, label=r'Theory ($N \rightarrow \infty$)'))
    
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_plot(results_dir, "losses.pdf", n_hidden, add_suffix=False)


def calculate_cosine_similarity(data):
    """Extract pre-computed cosine similarities from loaded data.
    
    Cosine similarities are computed and saved during training, so we just
    extract them from the loaded data dictionary.
    """
    similarities = {}
    for gamma_0 in data["gamma_0s"]:
        if gamma_0 in data["grad_cosine_similarities"]:
            similarities[gamma_0] = data["grad_cosine_similarities"][gamma_0]
    return similarities


def plot_cosine_similarity(data, similarities, results_dir, colormap_name='viridis', n_hidden=None):
    """Plot cosine similarity between BP grads and PC grads."""
    plot_gammas_series(
        {"gamma_0s": data["gamma_0s"], "cosine": similarities}, "cosine", results_dir,
        colormap_name, r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}^*\right)$",
        "grads_cosine_similarities.pdf", n_hidden
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning regimes results (varying gamma_0)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory containing experiment outputs"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2048,
        help="Width N to use for all plots"
    )
    parser.add_argument(
        "--gamma_0s",
        type=float,
        nargs='+',
        default=[0.1, 0.5, 1.0, 2.0, 3.0, 4.0],
        help="List of gamma_0 values to plot. If not provided, discovers all available gamma_0s."
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
        choices=["all", "rescalings", "losses", "energies", "ratio", "cosine"],
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
        "--n_hidden",
        type=int,
        default=4,
        help="Number of hidden layers H. If not provided, plots without n_hidden filtering."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="learning_regimes_plots",
        help="Directory to save plots. Will create subdirectories for dataset ID and then for each n_hidden."
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
    parser.add_argument(
        "--plot_theory",
        action="store_true",
        default=False,
        help="Whether to plot DMFT theory lines (default: False)"
    )
    parser.add_argument(
        "--param_type",
        type=str,
        default=None,
        help="Parameter type to filter results by ('sp', 'mupc', 'ntp'). If not provided, plots both 'sp' and 'mupc' in separate subfolders."
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
    
    args = parser.parse_args()
    
    # Extract dataset ID from results directory
    dataset_id = extract_dataset_id(args.results_dir)
    print(f"Detected dataset ID: {dataset_id}")
    
    # Determine which param types to process
    if args.param_type is not None:
        param_types = [args.param_type]
    else:
        # Default: process both sp and mupc
        param_types = ['sp', 'mupc']
    
    # Determine which use_skips values to process
    # Always iterate over both True and False unless explicitly specified
    if args.use_skips is not None:
        use_skips_values = [args.use_skips]
    else:
        # Default: automatically process both True and False
        use_skips_values = [True, False]
    
    # Create base plot directory
    base_plot_dir = os.path.join(args.results_dir, args.plot_dir)
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_plot_dir = os.path.join(base_plot_dir, dataset_id)
    os.makedirs(dataset_plot_dir, exist_ok=True)
    
    # Create width-specific subdirectory
    width_dir = os.path.join(dataset_plot_dir, f"{args.width}_width")
    os.makedirs(width_dir, exist_ok=True)
    
    # Create n_hidden-specific subdirectory (if specified)
    if args.n_hidden is not None:
        n_hidden_dir = os.path.join(width_dir, f"{args.n_hidden}_n_hidden")
    else:
        n_hidden_dir = os.path.join(width_dir, "all")
    os.makedirs(n_hidden_dir, exist_ok=True)
    
    # Process each param type
    for param_type in param_types:
        print(f"\n{'='*80}")
        print(f"Processing param_type: {param_type}")
        print(f"{'='*80}\n")
        
        # Process each use_skips configuration
        for use_skips in use_skips_values:
            print(f"\n{'='*70}")
            print(f"Processing use_skips: {use_skips}, param_type: {param_type}")
            print(f"{'='*70}\n")
            
            # Discover or use provided gamma_0s for this param_type and use_skips
            if args.gamma_0s is None:
                print(f"Discovering available gamma_0 values for width {args.width}...")
                print(f"  Filtering by param_type: {param_type}, use_skips: {use_skips}")
                gamma_0s = discover_available_gamma_0s(
                    args.results_dir, args.width, seed=args.seed, 
                    n_hidden=args.n_hidden, param_type=param_type, use_skips=use_skips
                )
                if not gamma_0s:
                    print(f"Warning: No gamma_0 values found for width {args.width} with param_type={param_type}, use_skips={use_skips}")
                    print(f"  Skipping {param_type}, use_skips={use_skips}...")
                    continue
                print(f"Found {len(gamma_0s)} gamma_0 values: {gamma_0s}")
            else:
                gamma_0s = args.gamma_0s
                print(f"Using provided gamma_0 values: {gamma_0s}")
            
            # Create nested subdirectories: use_skips -> param_type
            skip_status_dir = os.path.join(n_hidden_dir, f"{use_skips}_use_skips")
            os.makedirs(skip_status_dir, exist_ok=True)
            
            plot_dir = os.path.join(skip_status_dir, f"{param_type}_param_type")
            os.makedirs(plot_dir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Processing plots for width N = {args.width}")
            if args.n_hidden is not None:
                print(f"  H = {args.n_hidden}")
            print(f"  param_type = {param_type}")
            print(f"  use_skips = {use_skips}")
            print(f"  gamma_0 values: {gamma_0s}")
            print(f"{'='*60}\n")
            
            # Load data
            print(f"Loading data from {args.results_dir}...")
            print(f"  Filtering by param_type: {param_type}, use_skips: {use_skips}")
            data = load_data_from_dir(
                args.results_dir, gamma_0s, args.width, 
                seed=args.seed, n_hidden=args.n_hidden, param_type=param_type, use_skips=use_skips
            )
            
            # Print summary of loaded data
            print(f"\nLoaded data summary:")
            dmft_status = 'Yes' if (("dmft_losses" in data and data["dmft_losses"]) or data.get("dmft_loss") is not None) else 'No'
            print(f"  DMFT loss: {dmft_status}")
            if "dmft_losses" in data and data["dmft_losses"]:
                print(f"  DMFT loss loaded for gamma_0s: {sorted(data['dmft_losses'].keys())}")
            elif data.get("dmft_loss") is not None:
                print(f"  DMFT loss loaded (single value, will be used for all gamma_0s)")
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
                ("rescalings", plot_rescalings, [data, plot_dir, args.colormap, args.n_hidden, args.output_dim], "pc_rescaling.pdf"),
                ("energies", plot_losses_and_energies, [data, plot_dir, args.colormap, args.n_hidden, args.log_x_scale, args.plot_theory], "losses_and_energies.pdf"),
                ("losses", plot_losses, [data, plot_dir, args.colormap, args.n_hidden, args.log_x_scale, args.plot_theory], "losses.pdf"),
            ]
            
            for plot_type, plot_func, plot_args, filename in plot_configs:
                if args.plot == "all" or args.plot == plot_type:
                    print(f"Generating {plot_type} plot...")
                    plot_func(*plot_args)
                    print(f"  Saved to {os.path.join(plot_dir, filename)}")
            
            if args.plot == "all" or args.plot == "ratio":
                for loss_type, filename in [("pc", "pc_loss_energy_ratio.pdf"), ("bp", "bp_loss_energy_ratio.pdf")]:
                    print(f"Generating {loss_type} loss/energy ratio plot...")
                    plot_loss_energy_ratio(data, plot_dir, args.colormap, loss_type, args.n_hidden)
                    print(f"  Saved to {os.path.join(plot_dir, filename)}")
            
            if args.plot == "all" or args.plot == "cosine":
                print("Extracting cosine similarities...")
                similarities = calculate_cosine_similarity(data)
                if similarities:
                    print("Generating cosine similarity plot...")
                    plot_cosine_similarity(data, similarities, plot_dir, args.colormap, args.n_hidden)
                    print(f"  Saved to {os.path.join(plot_dir, 'grads_cosine_similarities.pdf')}")
                else:
                    print("  Warning: No cosine similarity data found")
            
            print(f"\nCompleted processing for param_type: {param_type}, use_skips: {use_skips}")
            print(f"  Plots saved to: {plot_dir}\n")
    
    print("\nDone!")
