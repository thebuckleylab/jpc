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

# Plot styling constants (matching plot_results.py)
FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
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


def load_energies_by_infer_mode(
    results_dir,
    widths,
    seed=0,
    n_hidden=None,
    param_type=None,
    use_skips=None,
    activity_lrs=None,
    act_fn=None,
):
    """Load energies for both closed_form and optim infer modes.
    
    Args:
        results_dir: Base results directory
        widths: List of widths to load
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        param_type: Parameter type to filter by ('sp' or 'mupc', None to load all)
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    
    Returns:
        Dictionary with structure:
        {
            "theory_energies": {width: array},  # from closed_form_infer_mode
            "simulation_energies": {width: array},  # from optim_infer_mode
            "widths": widths,
            "gamma_0": float or None
        }
    """
    data = {
        "theory_energies": {},
        "simulation_energies": {},
        "widths": widths,
        "gamma_0": None
    }
    
    # Build directory map
    dir_map = {}
    for root, _, files in os.walk(results_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            dir_map[root] = npy_files
    
    # Prepare optional filters
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    act_fn_str = f"{act_fn}_act_fn" if act_fn else None
    activity_lr_strs = (
        [f"{lr}_activity_lr" for lr in activity_lrs] if activity_lrs is not None else None
    )
    seed_str = str(seed)
    n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
    
    for width in widths:
        width_str = f"{width}_width"
        
        theory_dir = None
        simulation_dir = None
        
        for dir_path, files in dir_map.items():
            if width_str in dir_path and dir_path.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in dir_path:
                    # Filter by param_type if specified
                    if param_type_str is None or param_type_str in dir_path:
                        # Filter by use_skips if specified
                        if use_skips_str is None or use_skips_str in dir_path:
                            # Filter by act_fn if specified
                            if act_fn_str is None or act_fn_str in dir_path:
                                # Filter by activity_lr if specified
                                if activity_lr_strs is None or any(
                                    lr_str in dir_path for lr_str in activity_lr_strs
                                ):
                                    if "energies.npy" in files:
                                        # Check infer_mode
                                        if "closed_form_infer_mode" in dir_path:
                                            theory_dir = dir_path
                                        elif "optim_infer_mode" in dir_path:
                                            simulation_dir = dir_path
        
        # Load theory energies (closed_form)
        if theory_dir:
            if data["gamma_0"] is None:
                for part in theory_dir.split(os.sep):
                    if part.endswith("_gamma_0"):
                        try:
                            data["gamma_0"] = float(part.replace("_gamma_0", ""))
                            break
                        except ValueError:
                            pass
            
            arr = _load_npy_safe(os.path.join(theory_dir, "energies.npy"))
            if arr is not None:
                data["theory_energies"][width] = arr
        
        # Load simulation energies (optim)
        if simulation_dir:
            arr = _load_npy_safe(os.path.join(simulation_dir, "energies.npy"))
            if arr is not None:
                data["simulation_energies"][width] = arr
    
    return data


def setup_plot(xlabel, ylabel, log_scale=False):
    """Setup common plot styling."""
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(xlabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(ylabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=FONT_SIZES["legend"])
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    if log_scale:
        plt.yscale('log', base=10)


def save_plot(results_dir, filename, n_hidden=None, add_suffix=True):
    """Save plot with optional n_hidden suffix."""
    if n_hidden is not None and add_suffix:
        filename = filename.replace('.pdf', f'_H{n_hidden}.pdf')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight')
    plt.close()


def plot_energy_comparison(data, results_dir, param_type, n_hidden, colormap_name='viridis'):
    """Plot energy over training: theory vs simulation.
    
    Args:
        data: Dictionary from load_energies_by_infer_mode
        results_dir: Directory to save plot
        param_type: Parameter type ('sp' or 'mupc')
        n_hidden: Number of hidden layers
        colormap_name: Colormap name for width colors
    """
    plt.figure(figsize=FIG_SIZE)
    
    # Get widths that have both theory and simulation data
    theory_widths = set(data["theory_energies"].keys())
    sim_widths = set(data["simulation_energies"].keys())
    common_widths = sorted(list(theory_widths & sim_widths))
    
    if not common_widths:
        print(f"  Warning: No common widths found for param_type={param_type}, n_hidden={n_hidden}")
        plt.close()
        return
    
    colormap = plt.get_cmap(colormap_name)
    n_widths = len(common_widths)
    
    # Green color for simulation
    sim_color = (0.35, 0.75, 0.35)  # Medium green RGB
    
    # Plot theory and simulation for each width
    for idx, width in enumerate(common_widths):
        # Plot simulation (optim) - solid light green line (plot first, lower zorder)
        sim_energies = np.array(data["simulation_energies"][width]).flatten()
        sim_iterations = np.arange(1, len(sim_energies) + 1)
        plt.plot(sim_iterations, sim_energies, '-',
                alpha=ALPHA, linewidth=LINE_WIDTH, color=sim_color, zorder=1)
        
        # Plot theory (closed_form) - dashed black line (plot on top, higher zorder)
        theory_energies = np.array(data["theory_energies"][width]).flatten()
        theory_iterations = np.arange(1, len(theory_energies) + 1)
        plt.plot(theory_iterations, theory_energies, '--', 
                color='black', linewidth=4, alpha=0.8, zorder=2)
    
    # Add legend entries for theory and simulation (once)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='black', linewidth=4, alpha=0.8, label=r'Theory, $\mathcal{F}(\mathbf{z}^*)$'),
        Line2D([0], [0], linestyle='-', color=sim_color, linewidth=LINE_WIDTH, label=r'Simulation, $\mathcal{F}(\mathbf{z}_{T_{\text{max}}})$')
    ]
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"])
    
    setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=False)
    save_plot(results_dir, f"energy_comparison_{param_type}.pdf", n_hidden)


def plot_energy_comparison_combined(all_data_dict, results_dir, param_type, colormap_name='viridis'):
    """Plot energy over training: theory vs simulation for all hidden layers combined.
    
    Args:
        all_data_dict: Dictionary mapping n_hidden -> data from load_energies_by_infer_mode
        results_dir: Directory to save plot
        param_type: Parameter type ('sp' or 'mupc')
        colormap_name: Colormap name (not used, but kept for consistency)
    """
    plt.figure(figsize=(11, 6))
    
    # Get all n_hiddens and sort them
    n_hiddens = sorted([n for n in all_data_dict.keys() if all_data_dict[n] is not None])
    
    if not n_hiddens:
        print(f"  Warning: No data found for param_type={param_type}")
        plt.close()
        return
    
    # Use Greens colormap for simulation lines (same as cosine similarity plots)
    greens_colormap = plt.get_cmap('Greens')
    base_sim_color = (0.35, 0.75, 0.35)  # Original green for legend
    
    # Greyscale values for different L (similar to plot_learning_regimes)
    n_depths = len(n_hiddens)
    
    # Plot theory and simulation for each n_hidden
    for idx, n_hidden in enumerate(n_hiddens):
        data = all_data_dict[n_hidden]
        
        # Get widths that have both theory and simulation data
        theory_widths = set(data["theory_energies"].keys())
        sim_widths = set(data["simulation_energies"].keys())
        common_widths = sorted(list(theory_widths & sim_widths))
        
        if not common_widths:
            continue
        
        # Use the first available width
        width = common_widths[0]
        
        # Get shading value for this depth (same as L entries)
        grey_val = 0.8 - (idx / (n_depths - 1)) * 0.6 if n_depths > 1 else 0.5
        
        # Shade simulation line using Greens colormap (same as cosine similarity)
        color_val = get_color_val(idx, n_depths, 'Greens')
        shaded_sim_color = greens_colormap(color_val)
        
        # Shade theory line (use greyscale)
        theory_grey_color = (grey_val, grey_val, grey_val)
        
        # Plot simulation (optim) - solid shaded green line (plot first, lower zorder)
        sim_energies = np.array(data["simulation_energies"][width]).flatten()
        sim_iterations = np.arange(1, len(sim_energies) + 1)
        plt.plot(sim_iterations, sim_energies, '-',
                alpha=ALPHA, linewidth=LINE_WIDTH, color=shaded_sim_color, zorder=1)
        
        # Plot theory (closed_form) - dashed shaded greyscale line (plot on top, higher zorder)
        theory_energies = np.array(data["theory_energies"][width]).flatten()
        theory_iterations = np.arange(1, len(theory_energies) + 1)
        plt.plot(theory_iterations, theory_energies, '--', 
                color=theory_grey_color, linewidth=4, alpha=0.8, zorder=2)
    
    # Create legend: Theory, Simulation, then L values in greyscale
    from matplotlib.lines import Line2D
    # Use original black and green for legend entries
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='black', linewidth=4, alpha=0.8, label=r'Theory, $\mathcal{F}(\mathbf{z}^*)$'),
        Line2D([0], [0], linestyle='-', color=base_sim_color, linewidth=LINE_WIDTH, label=r'Sim., $\mathcal{F}(\mathbf{z}_{T_{\text{max}}})$')
    ]
    
    # Add L values in greyscale
    for idx, n_hidden in enumerate(n_hiddens):
        grey_val = 0.8 - (idx / (n_depths - 1)) * 0.6 if n_depths > 1 else 0.5
        grey_color = (grey_val, grey_val, grey_val)
        depth = n_hidden + 1
        legend_elements.append(Line2D([0], [0], color=grey_color, linestyle='-', linewidth=LINE_WIDTH, label=f'$L = {depth}$'))
    
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=False)
    save_plot(results_dir, f"energy_comparison_{param_type}_combined.pdf", None)


def plot_energy_delta(all_data_dict, results_dir, param_type, colormap_name='viridis'):
    """Plot absolute energy delta |theory - simulation| over training for different depths.
    
    Args:
        all_data_dict: Dictionary mapping n_hidden -> data from load_energies_by_infer_mode
        results_dir: Directory to save plot
        param_type: Parameter type ('sp' or 'mupc')
        colormap_name: Colormap name for depth colors
    """
    plt.figure(figsize=FIG_SIZE)
    
    # Get all n_hiddens and sort them
    n_hiddens = sorted([n for n in all_data_dict.keys() if all_data_dict[n] is not None])
    
    if not n_hiddens:
        print(f"  Warning: No data found for param_type={param_type}")
        plt.close()
        return
    
    colormap = plt.get_cmap(colormap_name)
    n_depths = len(n_hiddens)
    
    for idx, n_hidden in enumerate(n_hiddens):
        data = all_data_dict[n_hidden]
        
        # Get widths that have both theory and simulation data
        theory_widths = set(data["theory_energies"].keys())
        sim_widths = set(data["simulation_energies"].keys())
        common_widths = sorted(list(theory_widths & sim_widths))
        
        if not common_widths:
            continue
        
        # Use the first available width (or could average across widths)
        width = common_widths[0]
        
        theory_energies = np.array(data["theory_energies"][width]).flatten()
        sim_energies = np.array(data["simulation_energies"][width]).flatten()
        
        # Ensure same length
        min_len = min(len(theory_energies), len(sim_energies))
        theory_energies = theory_energies[:min_len]
        sim_energies = sim_energies[:min_len]
        
        # Calculate absolute delta
        energy_delta = np.abs(theory_energies - sim_energies)
        iterations = np.arange(1, len(energy_delta) + 1)
        
        color = colormap(0.15 + (idx / max(n_depths - 1, 1)) * 0.85 if n_depths > 1 else 0.15)
        
        # Use depth (n_hidden + 1) in legend
        depth = n_hidden + 1
        plt.plot(iterations, energy_delta, '-',
                label=f'$L = {depth}$',
                alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    # Add theory line at y=0 (matching energy comparison style, on top of other lines)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=4, alpha=0.8, zorder=10)
    
    # Create custom legend with Theory entry at the bottom
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    
    # Add L entries from existing plot labels first
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_handles.extend(handles)
    legend_labels.extend(labels)
    
    # Add Theory entry at the bottom of legend
    legend_handles.append(Line2D([0], [0], color='black', linewidth=4, linestyle='--', alpha=0.8))
    legend_labels.append('Theory')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$|\Delta \mathcal{F}(\boldsymbol{\theta}_t)|$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"])
    save_plot(results_dir, f"energy_delta_{param_type}.pdf", None)


def plot_energy_delta_combined(all_data_dict_sp, all_data_dict_mupc, results_dir, colormap_name='viridis'):
    """Plot absolute energy delta |theory - simulation| for both param types and all depths.
    
    Args:
        all_data_dict_sp: Dictionary mapping n_hidden -> data for SP param type
        all_data_dict_mupc: Dictionary mapping n_hidden -> data for muPC param type
        results_dir: Directory to save plot
        colormap_name: Colormap name (not used, but kept for consistency)
    """
    plt.figure(figsize=(10, 6))
    
    # Get all n_hiddens from both param types
    n_hiddens_sp = sorted([n for n in all_data_dict_sp.keys() if all_data_dict_sp[n] is not None])
    n_hiddens_mupc = sorted([n for n in all_data_dict_mupc.keys() if all_data_dict_mupc[n] is not None])
    all_n_hiddens = sorted(set(n_hiddens_sp + n_hiddens_mupc))
    
    if not all_n_hiddens:
        print(f"  Warning: No data found for combined delta plot")
        plt.close()
        return
    
    # Base colors: green for SP, purple for muPC (for legend)
    sp_base_color = np.array([0.0, 0.6, 0.0])  # Green
    mupc_base_color = np.array([0.6, 0.0, 0.6])  # Purple
    
    # Colormaps for shading
    greens_colormap = plt.get_cmap('Greens')
    purples_colormap = plt.get_cmap('Purples')
    
    # Greyscale values for different L (similar to plot_learning_regimes)
    n_depths = len(all_n_hiddens)
    
    # Plot SP lines
    for idx, n_hidden in enumerate(all_n_hiddens):
        if n_hidden not in all_data_dict_sp or all_data_dict_sp[n_hidden] is None:
            continue
        
        data = all_data_dict_sp[n_hidden]
        
        # Get widths that have both theory and simulation data
        theory_widths = set(data["theory_energies"].keys())
        sim_widths = set(data["simulation_energies"].keys())
        common_widths = sorted(list(theory_widths & sim_widths))
        
        if not common_widths:
            continue
        
        # Use the first available width
        width = common_widths[0]
        
        theory_energies = np.array(data["theory_energies"][width]).flatten()
        sim_energies = np.array(data["simulation_energies"][width]).flatten()
        
        # Ensure same length
        min_len = min(len(theory_energies), len(sim_energies))
        theory_energies = theory_energies[:min_len]
        sim_energies = sim_energies[:min_len]
        
        # Calculate absolute delta
        energy_delta = np.abs(theory_energies - sim_energies)
        iterations = np.arange(1, len(energy_delta) + 1)
        
        # Use Greens colormap: get color value from lighter to darker
        color_val = 0.15 + (idx / max(n_depths - 1, 1)) * 0.85 if n_depths > 1 else 0.15
        sp_color = greens_colormap(color_val)
        
        depth = n_hidden + 1
        plt.plot(iterations, energy_delta, '-',
                alpha=ALPHA, linewidth=LINE_WIDTH, color=sp_color)
    
    # Plot muPC lines
    for idx, n_hidden in enumerate(all_n_hiddens):
        if n_hidden not in all_data_dict_mupc or all_data_dict_mupc[n_hidden] is None:
            continue
        
        data = all_data_dict_mupc[n_hidden]
        
        # Get widths that have both theory and simulation data
        theory_widths = set(data["theory_energies"].keys())
        sim_widths = set(data["simulation_energies"].keys())
        common_widths = sorted(list(theory_widths & sim_widths))
        
        if not common_widths:
            continue
        
        # Use the first available width
        width = common_widths[0]
        
        theory_energies = np.array(data["theory_energies"][width]).flatten()
        sim_energies = np.array(data["simulation_energies"][width]).flatten()
        
        # Ensure same length
        min_len = min(len(theory_energies), len(sim_energies))
        theory_energies = theory_energies[:min_len]
        sim_energies = sim_energies[:min_len]
        
        # Calculate absolute delta
        energy_delta = np.abs(theory_energies - sim_energies)
        iterations = np.arange(1, len(energy_delta) + 1)
        
        # Use Purples colormap: get color value from lighter to darker
        color_val = 0.15 + (idx / max(n_depths - 1, 1)) * 0.85 if n_depths > 1 else 0.15
        mupc_color = purples_colormap(color_val)
        
        depth = n_hidden + 1
        plt.plot(iterations, energy_delta, '-',
                alpha=ALPHA, linewidth=LINE_WIDTH, color=mupc_color)
    
    # Add theory line at y=0 (matching energy comparison style, on top of other lines)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=4, alpha=0.8, zorder=10)
    
    # Create legend: SP (green), muPC (purple), then L values in greyscale, then Theory at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=tuple(sp_base_color), linestyle='-', linewidth=LINE_WIDTH, label='SP'),
        Line2D([0], [0], color=tuple(mupc_base_color), linestyle='-', linewidth=LINE_WIDTH, label='$\mu$P')
    ]
    
    # Add L values in greyscale
    for idx, n_hidden in enumerate(all_n_hiddens):
        grey_val = 0.8 - (idx / (n_depths - 1)) * 0.6 if n_depths > 1 else 0.5
        grey_color = (grey_val, grey_val, grey_val)
        depth = n_hidden + 1
        legend_elements.append(Line2D([0], [0], color=grey_color, linestyle='-', linewidth=LINE_WIDTH, label=f'$L = {depth}$'))
    
    # Add Theory entry at the bottom of legend
    legend_elements.append(Line2D([0], [0], color='black', linewidth=4, linestyle='--', alpha=0.8, label='Theory'))
    
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    setup_plot("$t$", r"$|\Delta \mathcal{F}(\boldsymbol{\theta}_t)|$", log_scale=False)
    save_plot(results_dir, "energy_delta_combined.pdf", None)


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


def load_data_by_infer_mode(
    results_dir,
    widths,
    seed=0,
    n_hidden=None,
    param_type=None,
    infer_mode=None,
    use_skips=None,
    activity_lrs=None,
    act_fn=None,
):
    """Load energies, train losses, BP losses, and cosine similarities for a specific infer_mode.
    
    Args:
        results_dir: Base results directory
        widths: List of widths to load
        seed: Seed value to match
        n_hidden: Number of hidden layers to match (None for any)
        param_type: Parameter type to filter by ('sp' or 'mupc')
        infer_mode: Infer mode to filter by ('closed_form' or 'optim')
        use_skips: Whether to filter by use_skips (True/False, None to load all)
    
    Returns:
        Dictionary with structure:
        {
            "energies": {width: array},
            "train_losses": {width: array},  # PC train losses
            "bp_losses": {width: array},     # BP losses (independent of infer_mode)
            "cosine_similarities": {width: array},
            "widths": widths
        }
    """
    data = {
        "energies": {},
        "train_losses": {},
        "bp_losses": {},
        "cosine_similarities": {},
        "widths": widths
    }
    
    # Build directory map
    dir_map = {}
    for root, _, files in os.walk(results_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            dir_map[root] = npy_files
    
    # Prepare optional filters
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    infer_mode_str = f"{infer_mode}_infer_mode" if infer_mode else None
    act_fn_str = f"{act_fn}_act_fn" if act_fn else None
    # activity_lr only appears in PC experiment directories, not BP ones
    activity_lr_strs = (
        [f"{lr}_activity_lr" for lr in activity_lrs] if activity_lrs is not None else None
    )
    seed_str = str(seed)
    n_hidden_str = f"{n_hidden}_n_hidden" if n_hidden else None
    
    for width in widths:
        width_str = f"{width}_width"
        
        # Find PC directory (with infer_mode)
        pc_dir = None
        # Find BP directory (without infer_mode, has optim_id instead)
        bp_dir = None
        
        for dir_path, files in dir_map.items():
            if width_str in dir_path and dir_path.split(os.sep)[-1] == seed_str:
                if n_hidden_str is None or n_hidden_str in dir_path:
                    # Filter by param_type if specified
                    if param_type_str is None or param_type_str in dir_path:
                        # Filter by use_skips if specified
                        if use_skips_str is None or use_skips_str in dir_path:
                            # Filter by act_fn if specified
                            if act_fn_str is None or act_fn_str in dir_path:
                                # PC dirs: have infer_mode; also encode activity_lr
                                if infer_mode_str and infer_mode_str in dir_path:
                                    if activity_lr_strs is None or any(
                                        lr_str in dir_path for lr_str in activity_lr_strs
                                    ):
                                        if "energies.npy" in files:
                                            pc_dir = dir_path
                                # BP dirs: have optim_id, no infer_mode, no activity_lr filter
                                elif (
                                    "optim_id" in dir_path
                                    and "infer_mode" not in dir_path
                                    and "losses.npy" in files
                                ):
                                    bp_dir = dir_path
        
        # Load data from PC directory
        if pc_dir:
            # Load energies
            arr = _load_npy_safe(os.path.join(pc_dir, "energies.npy"))
            if arr is not None:
                data["energies"][width] = arr
            
            # Load PC train losses
            arr = _load_npy_safe(os.path.join(pc_dir, "train_losses.npy"))
            if arr is not None:
                data["train_losses"][width] = arr
            
            # Load cosine similarities
            arr = _load_npy_safe(os.path.join(pc_dir, "grad_cosine_similarities.npy"))
            if arr is not None:
                data["cosine_similarities"][width] = arr
        
        # Load BP losses (independent of infer_mode, so load once per width)
        if bp_dir and width not in data["bp_losses"]:
            arr = _load_npy_safe(os.path.join(bp_dir, "losses.npy"))
            if arr is not None:
                data["bp_losses"][width] = arr
    
    return data


def plot_losses_and_energies(all_data_dict, results_dir, param_type, infer_mode, colormap_name='viridis'):
    """Plot energies and train losses for different depths L.
    
    Args:
        all_data_dict: Dictionary mapping n_hidden -> data from load_data_by_infer_mode
        results_dir: Directory to save plot
        param_type: Parameter type ('sp' or 'mupc')
        infer_mode: Infer mode ('closed_form' or 'optim')
        colormap_name: Colormap name (not used, kept for consistency)
    """
    plt.figure(figsize=(11, 6))
    
    # Get all n_hiddens and sort them
    n_hiddens = sorted([n for n in all_data_dict.keys() if all_data_dict[n] is not None])
    
    if not n_hiddens:
        print(f"  Warning: No data found for param_type={param_type}, infer_mode={infer_mode}")
        plt.close()
        return
    
    # Use Blues for energy, Reds for loss
    blues_colormap = plt.get_cmap('Blues')
    reds_colormap = plt.get_cmap('Reds')
    n_depths = len(n_hiddens)
    
    # Get medium colors for PC and BP
    pc_middle_color = blues_colormap(0.5)
    bp_middle_color = reds_colormap(0.5)
    
    # Plot energies for each depth (using Blues)
    for idx, n_hidden in enumerate(n_hiddens):
        data = all_data_dict[n_hidden]
        # Use first available width
        widths_with_energies = sorted([w for w in data["widths"] if w in data["energies"]])
        if widths_with_energies:
            width = widths_with_energies[0]
            energies = np.array(data["energies"][width]).flatten()
            iterations = np.arange(1, len(energies) + 1)
            color = blues_colormap(get_color_val(idx, n_depths, 'Blues'))
            plt.plot(iterations, energies, '-',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    # Plot BP losses for each depth (using Reds) - BP losses are independent of infer_mode
    for idx, n_hidden in enumerate(n_hiddens):
        data = all_data_dict[n_hidden]
        # Use first available width
        widths_with_bp_losses = sorted([w for w in data["widths"] if w in data["bp_losses"]])
        if widths_with_bp_losses:
            width = widths_with_bp_losses[0]
            losses = np.array(data["bp_losses"][width]).flatten()
            iterations = np.arange(1, len(losses) + 1)
            color = reds_colormap(get_color_val(idx, n_depths, 'Reds'))
            plt.plot(iterations, losses, '--',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    # Create legend: PC (medium blue), BP (medium red), then L values in greyscale
    from matplotlib.lines import Line2D
    # Set PC label based on infer_mode
    if infer_mode == "closed_form":
        pc_label = r'$\mathcal{F}(\mathbf{z}^*)$ (PC)'
    else:  # optim
        pc_label = r'$\mathcal{F}(\mathbf{z}_{T_{\text{max}}})$ (PC)'
    
    legend_elements = [
        Line2D([0], [0], color=pc_middle_color, linestyle='-', linewidth=LINE_WIDTH, label=pc_label),
        Line2D([0], [0], color=bp_middle_color, linestyle='--', linewidth=LINE_WIDTH, label=r'$\mathcal{L}(\boldsymbol{\theta})$ (BP)')
    ]
    
    # Add L values in greyscale
    for idx, n_hidden in enumerate(n_hiddens):
        grey_val = 0.8 - (idx / (n_depths - 1)) * 0.6 if n_depths > 1 else 0.5
        grey_color = (grey_val, grey_val, grey_val)
        depth = n_hidden + 1
        legend_elements.append(Line2D([0], [0], color=grey_color, linestyle='-', linewidth=LINE_WIDTH, label=f'$L = {depth}$'))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", 
               fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    plt.yscale('log', base=10)
    
    filename = f"losses_and_energies_{param_type}_{infer_mode}.pdf"
    save_plot(results_dir, filename, None, add_suffix=False)


def plot_cosine_similarity(all_data_dict, results_dir, param_type, infer_mode, colormap_name='viridis'):
    """Plot cosine similarity between BP grads and PC grads for different depths L.
    
    Args:
        all_data_dict: Dictionary mapping n_hidden -> data from load_data_by_infer_mode
        results_dir: Directory to save plot
        param_type: Parameter type ('sp' or 'mupc')
        infer_mode: Infer mode ('closed_form' or 'optim')
        colormap_name: Colormap name
    """
    plt.figure(figsize=FIG_SIZE)
    
    # Get all n_hiddens and sort them
    n_hiddens = sorted([n for n in all_data_dict.keys() if all_data_dict[n] is not None])
    
    if not n_hiddens:
        print(f"  Warning: No data found for param_type={param_type}, infer_mode={infer_mode}")
        plt.close()
        return
    
    colormap = plt.get_cmap(colormap_name)
    n_depths = len(n_hiddens)
    
    for idx, n_hidden in enumerate(n_hiddens):
        data = all_data_dict[n_hidden]
        # Use first available width
        widths_with_cosine = sorted([w for w in data["widths"] if w in data["cosine_similarities"]])
        if widths_with_cosine:
            width = widths_with_cosine[0]
            values = np.array(data["cosine_similarities"][width]).flatten()
            iterations = np.arange(1, len(values) + 1)
            color = colormap(get_color_val(idx, n_depths, colormap_name))
            depth = n_hidden + 1
            plt.plot(iterations, values, label=f'$L = {depth}$',
                    alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
    
    # Add theory line at y=1 for mupc optim (matching energy delta style, on top of other lines)
    if param_type == "mupc" and infer_mode == "optim":
        plt.axhline(y=1, color='black', linestyle='--', linewidth=4, alpha=0.8, zorder=10)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    # Remove star from F for optim infer_mode
    if infer_mode == "optim":
        ylabel = r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$"
    else:
        ylabel = r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}^*\right)$"
    plt.ylabel(ylabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    
    # Create legend with Theory entry at the bottom for mupc optim
    from matplotlib.lines import Line2D
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        legend_handles = handles
        legend_labels = labels
        # Add Theory entry at the bottom for mupc optim
        if param_type == "mupc" and infer_mode == "optim":
            legend_handles.append(Line2D([0], [0], color='black', linewidth=4, linestyle='--', alpha=0.8))
            legend_labels.append('Theory')
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.0, 0.0), loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    
    # Set reasonable y-axis limits only for muPC closed_form (values are very close to 1)
    if param_type == "mupc" and infer_mode == "closed_form":
        all_values = []
        for n_hidden in n_hiddens:
            data = all_data_dict[n_hidden]
            widths_with_cosine = sorted([w for w in data["widths"] if w in data["cosine_similarities"]])
            if widths_with_cosine:
                width = widths_with_cosine[0]
                values = np.array(data["cosine_similarities"][width]).flatten()
                all_values.extend(values.tolist())
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            # If values are close to 1, use a tighter range; otherwise use full range
            if max_val > 0.9:
                plt.ylim(max(0.8, min_val - 0.05), min(1.05, max_val + 0.05))
            else:
                plt.ylim(max(-1.05, min_val - 0.05), min(1.05, max_val + 0.05))
    
    filename = f"cosine_similarity_{param_type}_{infer_mode}.pdf"
    save_plot(results_dir, filename, None, add_suffix=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot inference convergence results")
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
        default=[2048],
        help="List of widths N to plot"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed value for loading data"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="Greens",
        help="Colormap to use for plotting"
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        nargs='+',
        default=[1, 3, 7],  # 1, 2, 3, 4
        help="List of hidden layer counts H to plot"
    )
    parser.add_argument(
        "--param_types",
        type=str,
        nargs='+',
        default=["sp", "mupc"],
        help="Parameter types to plot"
    )
    parser.add_argument(
        "--act_fn",
        type=str,
        default="linear",
        help="Activation function to filter results by (e.g. 'tanh', 'relu'). If not set, use all available act_fns."
    )
    parser.add_argument(
        "--activity_lrs",
        type=float,
        nargs='+',
        default=[1],
        help="Activity learning rates to include in the plots. If not set, use all available activity_lr values."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="infer_convergence_plots",
        help="Directory to save plots"
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
    
    # Determine which use_skips values to process
    if args.use_skips is not None:
        use_skips_values = [args.use_skips]
    else:
        use_skips_values = [True, False]
    
    # Create base plot directory
    base_plot_dir = os.path.join(args.results_dir, args.plot_dir)
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # Process each use_skips configuration
    for use_skips in use_skips_values:
        print(f"\n{'='*80}")
        print(f"Processing use_skips = {use_skips}")
        print(f"{'='*80}\n")
        
        # Create subdirectory for this skip status
        skip_status_dir = os.path.join(base_plot_dir, f"{use_skips}_use_skips")
        os.makedirs(skip_status_dir, exist_ok=True)
        
        # Store data for both param types for combined plot
        all_data_dict_sp = {}
        all_data_dict_mupc = {}
        
        # Process each param_type
        for param_type in args.param_types:
            print(f"\n{'='*60}")
            print(f"Processing param_type = {param_type}, use_skips = {use_skips}")
            print(f"{'='*60}\n")
            
            # Store all data for delta plot (across all n_hiddens)
            all_data_dict = {}
            
            # First pass: load data and plot energy comparison for each n_hidden
            for n_hidden in args.n_hiddens:
                print(f"  Processing n_hidden = {n_hidden}")
                
                # Load data for this configuration
                data = load_energies_by_infer_mode(
                    args.results_dir,
                    args.widths,
                    seed=args.seed,
                    n_hidden=n_hidden,
                    param_type=param_type,
                    use_skips=use_skips,
                    activity_lrs=args.activity_lrs,
                    act_fn=args.act_fn,
                )
                
                # Check if we have data
                if not data["theory_energies"] or not data["simulation_energies"]:
                    print(f"    Warning: Missing data for param_type={param_type}, n_hidden={n_hidden}, use_skips={use_skips}")
                    print(f"      Theory energies: {sorted(data['theory_energies'].keys())}")
                    print(f"      Simulation energies: {sorted(data['simulation_energies'].keys())}")
                    all_data_dict[n_hidden] = None
                    continue
                
                all_data_dict[n_hidden] = data
                
                # Store for combined plot
                if param_type == "sp":
                    all_data_dict_sp[n_hidden] = data
                elif param_type == "mupc":
                    all_data_dict_mupc[n_hidden] = data
                
                # Create subdirectory for this n_hidden
                plot_subdir = os.path.join(skip_status_dir, f"{n_hidden}_n_hidden")
                os.makedirs(plot_subdir, exist_ok=True)
                
                # Plot energy comparison (per n_hidden)
                print(f"    Generating energy comparison plot...")
                plot_energy_comparison(data, plot_subdir, param_type, n_hidden, args.colormap)
                print(f"      Saved to {os.path.join(plot_subdir, f'energy_comparison_{param_type}_H{n_hidden}.pdf')}")
            
            # Generate combined energy comparison plot (all depths)
            print(f"  Generating combined energy comparison plot (all depths)...")
            plot_energy_comparison_combined(all_data_dict, skip_status_dir, param_type, args.colormap)
            print(f"    Saved to {os.path.join(skip_status_dir, f'energy_comparison_{param_type}_combined.pdf')}")
            
            # Second pass: plot energy delta across all n_hiddens
            print(f"  Generating energy delta plot (all depths)...")
            plot_energy_delta(all_data_dict, skip_status_dir, param_type, args.colormap)
            print(f"    Saved to {os.path.join(skip_status_dir, f'energy_delta_{param_type}.pdf')}")
        
        # Generate combined plot for both param types
        if all_data_dict_sp and all_data_dict_mupc:
            print(f"\n{'='*60}")
            print(f"Generating combined energy delta plot (both param types)")
            print(f"{'='*60}\n")
            plot_energy_delta_combined(all_data_dict_sp, all_data_dict_mupc, skip_status_dir, args.colormap)
            print(f"  Saved to {os.path.join(skip_status_dir, 'energy_delta_combined.pdf')}")
        
        # Generate losses and energies plots for each param_type and infer_mode combination
        # 4 plots total: SP closed_form, SP optim, muPC closed_form, muPC optim
        print(f"\n{'='*60}")
        print(f"Generating losses and energies plots")
        print(f"{'='*60}\n")
        for param_type in args.param_types:
            for infer_mode in ["closed_form", "optim"]:
                print(f"  Processing {param_type} with {infer_mode} infer_mode...")
                # Load data for all n_hiddens
                all_data_dict = {}
                
                for n_hidden in args.n_hiddens:
                    data = load_data_by_infer_mode(
                        args.results_dir,
                        args.widths,
                        seed=args.seed,
                        n_hidden=n_hidden,
                        param_type=param_type,
                        infer_mode=infer_mode,
                        use_skips=use_skips,
                        activity_lrs=args.activity_lrs,
                        act_fn=args.act_fn,
                    )
                    
                    if data["energies"] or data["train_losses"]:
                        all_data_dict[n_hidden] = data
                
                if all_data_dict:
                    plot_losses_and_energies(all_data_dict, skip_status_dir, param_type, infer_mode, args.colormap)
                    print(f"    Saved to {os.path.join(skip_status_dir, f'losses_and_energies_{param_type}_{infer_mode}.pdf')}")
        
        # Generate cosine similarity plots for each param_type and infer_mode combination
        # 4 plots total: SP closed_form, SP optim, muPC closed_form, muPC optim
        print(f"\n{'='*60}")
        print(f"Generating cosine similarity plots")
        print(f"{'='*60}\n")
        for param_type in args.param_types:
            for infer_mode in ["closed_form", "optim"]:
                print(f"  Processing {param_type} with {infer_mode} infer_mode...")
                # Load data for all n_hiddens
                all_data_dict = {}
                
                for n_hidden in args.n_hiddens:
                    data = load_data_by_infer_mode(
                        args.results_dir,
                        args.widths,
                        seed=args.seed,
                        n_hidden=n_hidden,
                        param_type=param_type,
                        infer_mode=infer_mode,
                        use_skips=use_skips,
                        activity_lrs=args.activity_lrs,
                        act_fn=args.act_fn,
                    )
                    
                    if data["cosine_similarities"]:
                        all_data_dict[n_hidden] = data
                
                if all_data_dict:
                    plot_cosine_similarity(all_data_dict, skip_status_dir, param_type, infer_mode, args.colormap)
                    print(f"    Saved to {os.path.join(skip_status_dir, f'cosine_similarity_{param_type}_{infer_mode}.pdf')}")
        
        print(f"\nCompleted processing for use_skips = {use_skips}")
    
    print("\nDone!")

