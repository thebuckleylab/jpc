import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_results import (
    extract_dataset_id,
    load_data_from_dir,
    get_color_val,
    is_sequential_colormap
)

# Plot styling constants (matching plot_main_results.py exactly)
FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7


def discover_available_param_types(results_dir, seed=0):
    """Discover all available param_types from the results directory.
    
    Args:
        results_dir: Directory to search
        seed: Seed value to match
    
    Returns:
        Set of param_type strings (e.g., {'sp', 'mupc'})
    """
    param_types = set()
    seed_str = str(seed)
    
    for root, _, files in os.walk(results_dir):
        if "loss_rescalings.npy" in files or "energies.npy" in files:
            if root.split(os.sep)[-1] == seed_str:
                for part in root.split(os.sep):
                    if part.endswith("_param_type"):
                        param_type = part.replace("_param_type", "")
                        param_types.add(param_type)
    
    return param_types


def discover_available_use_skips(results_dir, seed=0, param_type=None):
    """Discover all available use_skips values from the results directory.
    
    Args:
        results_dir: Directory to search
        seed: Seed value to match
        param_type: Parameter type to filter by (None to check all)
    
    Returns:
        Set of use_skips values (True, False, or both)
    """
    use_skips_values = set()
    seed_str = str(seed)
    param_type_str = f"{param_type}_param_type" if param_type else None
    
    for root, _, files in os.walk(results_dir):
        if "loss_rescalings.npy" in files or "energies.npy" in files:
            if root.split(os.sep)[-1] == seed_str:
                if param_type_str is None or param_type_str in root:
                    for part in root.split(os.sep):
                        if part.endswith("_use_skips"):
                            use_skips_str = part.replace("_use_skips", "")
                            if use_skips_str.lower() == "true":
                                use_skips_values.add(True)
                            elif use_skips_str.lower() == "false":
                                use_skips_values.add(False)
    
    return use_skips_values


def plot_losses(data_by_n_hidden, results_dir, colormap_name='viridis', width=None, log_xaxis=True, n_hiddens=None, max_iterations=None):
    """Plot PC train loss and BP loss at different L values, matching plot_main_results style exactly."""
    plt.figure(figsize=(10, 6))
    
    # Get all available n_hiddens
    if n_hiddens is None:
        n_hiddens = sorted([n for n in data_by_n_hidden.keys() if n is not None])
    
    # Get all available n_hiddens for both PC and BP
    pc_n_hiddens_list = sorted([n for n in n_hiddens if n in data_by_n_hidden and data_by_n_hidden[n].get("pc_train_losses")])
    bp_n_hiddens_list = sorted([n for n in n_hiddens if n in data_by_n_hidden and data_by_n_hidden[n].get("bp_losses")])
    all_n_hiddens = sorted(set(pc_n_hiddens_list + bp_n_hiddens_list))
    
    # Middle blue color for PC legend entry (matching plot_main_results)
    pc_legend_color = '#4A90E2'
    bp_color = '#DC143C'
    
    # Plot PC train losses with Blues colormap (no labels)
    if pc_n_hiddens_list and width is not None:
        blues_cmap = plt.get_cmap('Blues')
        n_L_values = len(pc_n_hiddens_list)
        for idx, n_hidden in enumerate(pc_n_hiddens_list):
            data = data_by_n_hidden[n_hidden]
            if width in data.get("pc_train_losses", {}):
                train_losses = np.array(data["pc_train_losses"][width]).flatten()
                if max_iterations is not None:
                    train_losses = train_losses[:max_iterations]
                iterations = np.arange(1, len(train_losses) + 1)
                color = blues_cmap(get_color_val(idx, n_L_values, 'Blues'))
                plt.plot(iterations, train_losses, '-', alpha=ALPHA, linewidth=LINE_WIDTH,
                        color=color, label='')
    
    # Plot BP losses with Reds colormap (no labels) - similar to PC but in red
    if bp_n_hiddens_list and width is not None:
        reds_cmap = plt.get_cmap('Reds')
        n_L_values = len(bp_n_hiddens_list)
        for idx, n_hidden in enumerate(bp_n_hiddens_list):
            data = data_by_n_hidden[n_hidden]
            if width in data.get("bp_losses", {}):
                bp_loss = np.array(data["bp_losses"][width]).flatten()
                if max_iterations is not None:
                    bp_loss = bp_loss[:max_iterations]
                iterations = np.arange(1, len(bp_loss) + 1)
                color = reds_cmap(get_color_val(idx, n_L_values, 'Reds'))
                plt.plot(iterations, bp_loss, '-', alpha=ALPHA, linewidth=LINE_WIDTH,
                        color=color, label='')
    
    # Create custom legend (matching plot_main_results exactly)
    legend_handles = []
    legend_labels = []
    
    # Add PC entry
    if pc_n_hiddens_list:
        legend_handles.append(Line2D([0], [0], color=pc_legend_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'PC')
    
    # Add BP entry
    if bp_n_hiddens_list:
        legend_handles.append(Line2D([0], [0], color=bp_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'BP')
    
    # Add L entries with grayscale (matching plot_main_results style)
    gray_cmap = plt.get_cmap('Greys')
    all_L_values = sorted([(n_hidden + 1) if n_hidden is not None else 1 for n_hidden in all_n_hiddens])
    n_all_L = len(all_L_values)
    for idx, L in enumerate(all_L_values):
        gray_val = 0.3 + (idx / max(n_all_L - 1, 1)) * 0.5 if n_all_L > 1 else 0.5
        gray_color = gray_cmap(gray_val)
        legend_handles.append(Line2D([0], [0], color=gray_color, linewidth=LINE_WIDTH))
        legend_labels.append(f'$L = {L}$')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if width is not None:
        plt.title(f"$N = {width}$", fontsize=FONT_SIZES["label"], pad=LABEL_PAD)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_xaxis:
        plt.xscale('log', base=10)
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    plt.yscale('log', base=10)
    plt.tight_layout()
    filename = f"losses_N{width}.pdf" if width is not None else "losses.pdf"
    plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight')
    plt.close()


def plot_losses_and_energies(data_by_n_hidden, results_dir, colormap_name='viridis', width=None, log_xaxis=True, n_hiddens=None, max_iterations=None):
    """Plot PC theory energy and BP loss at different L values, matching plot_main_results style exactly."""
    plt.figure(figsize=(10, 6))
    
    # Get all available n_hiddens
    if n_hiddens is None:
        n_hiddens = sorted([n for n in data_by_n_hidden.keys() if n is not None])
    
    # Get all available n_hiddens for both PC and BP
    pc_n_hiddens_list = sorted([n for n in n_hiddens if n in data_by_n_hidden and data_by_n_hidden[n].get("pc_energies")])
    bp_n_hiddens_list = sorted([n for n in n_hiddens if n in data_by_n_hidden and data_by_n_hidden[n].get("bp_losses")])
    all_n_hiddens = sorted(set(pc_n_hiddens_list + bp_n_hiddens_list))
    
    # Middle blue color for PC legend entry (matching plot_main_results)
    pc_legend_color = '#4A90E2'
    bp_color = '#DC143C'
    
    # Plot PC energies with Blues colormap (no labels)
    if pc_n_hiddens_list and width is not None:
        blues_cmap = plt.get_cmap('Blues')
        n_L_values = len(pc_n_hiddens_list)
        for idx, n_hidden in enumerate(pc_n_hiddens_list):
            data = data_by_n_hidden[n_hidden]
            if width in data.get("pc_energies", {}):
                energies = np.array(data["pc_energies"][width]).flatten()
                if max_iterations is not None:
                    energies = energies[:max_iterations]
                iterations = np.arange(1, len(energies) + 1)
                color = blues_cmap(get_color_val(idx, n_L_values, 'Blues'))
                plt.plot(iterations, energies, '-', alpha=ALPHA, linewidth=LINE_WIDTH,
                        color=color, label='')
    
    # Plot BP losses with Reds colormap (no labels) - similar to PC but in red
    if bp_n_hiddens_list and width is not None:
        reds_cmap = plt.get_cmap('Reds')
        n_L_values = len(bp_n_hiddens_list)
        for idx, n_hidden in enumerate(bp_n_hiddens_list):
            data = data_by_n_hidden[n_hidden]
            if width in data.get("bp_losses", {}):
                bp_loss = np.array(data["bp_losses"][width]).flatten()
                if max_iterations is not None:
                    bp_loss = bp_loss[:max_iterations]
                iterations = np.arange(1, len(bp_loss) + 1)
                color = reds_cmap(get_color_val(idx, n_L_values, 'Reds'))
                plt.plot(iterations, bp_loss, '-', alpha=ALPHA, linewidth=LINE_WIDTH,
                        color=color, label='')
    
    # Create custom legend (matching plot_main_results exactly)
    legend_handles = []
    legend_labels = []
    
    # Add PC entry
    if pc_n_hiddens_list:
        legend_handles.append(Line2D([0], [0], color=pc_legend_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'$\mathcal{F}^*(\boldsymbol{\theta})$ (PC)')
    
    # Add BP entry
    if bp_n_hiddens_list:
        legend_handles.append(Line2D([0], [0], color=bp_color, linewidth=LINE_WIDTH, alpha=ALPHA))
        legend_labels.append(r'$\mathcal{L}(\boldsymbol{\theta})$ (BP)')
    
    # Add L entries with grayscale (matching plot_main_results style)
    gray_cmap = plt.get_cmap('Greys')
    all_L_values = sorted([(n_hidden + 1) if n_hidden is not None else 1 for n_hidden in all_n_hiddens])
    n_all_L = len(all_L_values)
    for idx, L in enumerate(all_L_values):
        gray_val = 0.3 + (idx / max(n_all_L - 1, 1)) * 0.5 if n_all_L > 1 else 0.5
        gray_color = gray_cmap(gray_val)
        legend_handles.append(Line2D([0], [0], color=gray_color, linewidth=LINE_WIDTH))
        legend_labels.append(f'$L = {L}$')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if width is not None:
        plt.title(f"$N = {width}$", fontsize=FONT_SIZES["label"], pad=LABEL_PAD)
    plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    if log_xaxis:
        plt.xscale('log', base=10)
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    plt.tight_layout()
    filename = f"losses_and_energies_N{width}.pdf" if width is not None else "losses_and_energies.pdf"
    plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling limits results for saddle regime")
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
        default=[1, 2, 4, 8],
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
        choices=["all", "losses", "energies", "cosine"],
        default="all",
        help="Which plot(s) to generate (grads excluded for sp)"
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
        default=[1, 3, 5, 7],
        help="List of hidden layer counts H to plot. If not provided, plots without n_hidden filtering."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="saddle_regime_plots",
        help="Directory to save plots. Will create subdirectories for dataset ID, param_type, use_skips, and then for each n_hidden."
    )
    parser.add_argument(
        "--gamma_0",
        type=float,
        default=1,
        help="Gamma_0 value for theoretical line and folder naming. If not provided, will be extracted from data directory."
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=1,
        help="Output dimension. If > 1, labels will use spectral norm notation (default: 1)"
    )
    parser.add_argument(
        "--log_xaxis",
        default=False,
        help="Use log scale on x-axis (default: True)"
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
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum number of training iterations to plot. If not provided, plots all available iterations."
    )
    
    args = parser.parse_args()
    
    # Extract dataset ID from results directory
    dataset_id = extract_dataset_id(args.results_dir)
    print(f"Detected dataset ID: {dataset_id}")
    
    # Only plot 'sp' param_type (mupc is excluded)
    param_types = ['sp']
    print(f"Using param_type: sp (mupc excluded)")
    
    # Determine which use_skips values to process
    if args.use_skips is not None:
        use_skips_values = [args.use_skips]
    else:
        # Discover available use_skips values for each param_type
        all_use_skips = set()
        for param_type in param_types:
            use_skips_for_type = discover_available_use_skips(args.results_dir, seed=args.seed, param_type=param_type)
            all_use_skips.update(use_skips_for_type)
        
        if not all_use_skips:
            # If none found, default to both True and False
            use_skips_values = [True, False]
            print("No use_skips values found. Defaulting to both True and False.")
        else:
            use_skips_values = sorted(list(all_use_skips), reverse=True)  # True before False
            print(f"Found use_skips values: {use_skips_values}")
    
    # Set n_hiddens to process (used for loading data, not for directory structure)
    if args.n_hiddens is None:
        n_hiddens = [None]
        print("No n_hiddens specified. Proceeding without n_hidden filtering.")
    else:
        n_hiddens = args.n_hiddens
        print(f"Processing plots for n_hiddens: {n_hiddens}")
    
    # Create base plot directory
    base_plot_dir = os.path.join(args.results_dir, args.plot_dir)
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_plot_dir = os.path.join(base_plot_dir, dataset_id)
    os.makedirs(dataset_plot_dir, exist_ok=True)
    
    # Loop over each param_type
    for param_type in param_types:
        print(f"\n{'='*80}")
        print(f"Processing param_type: {param_type}")
        print(f"{'='*80}\n")
        
        # Create param_type subdirectory
        param_type_plot_dir = os.path.join(dataset_plot_dir, param_type)
        os.makedirs(param_type_plot_dir, exist_ok=True)
        
        # Loop over each use_skips configuration
        for use_skips in use_skips_values:
            print(f"\n{'='*60}")
            print(f"Processing use_skips = {use_skips}")
            print(f"{'='*60}\n")
            
            # Create subdirectory for this skip status
            skip_status_dir = os.path.join(param_type_plot_dir, f"{use_skips}_use_skips")
            os.makedirs(skip_status_dir, exist_ok=True)
            
            # Loop over each width (N) configuration
            for width in args.widths:
                print(f"\n{'='*60}")
                print(f"Processing plots for N = {width}")
                print(f"{'='*60}\n")
                
                # Create subdirectory for this width within the skip status subdirectory
                plot_dir = os.path.join(skip_status_dir, f"N{width}")
                os.makedirs(plot_dir, exist_ok=True)
                
                # Load data for each n_hidden separately (for this fixed width)
                data_by_n_hidden = {}
                gamma_0 = args.gamma_0
                
                print(f"Loading data from {args.results_dir} for width N = {width}...")
                for n_hidden in n_hiddens:
                    data = load_data_from_dir(
                        args.results_dir, 
                        [width],  # Only load data for this specific width
                        seed=args.seed, 
                        n_hidden=n_hidden,
                        param_type=param_type,
                        use_skips=use_skips
                    )
                    
                    # Store data keyed by n_hidden if it contains data for this width
                    has_data = False
                    if data:
                        # Check if data contains the width key in any of the data dictionaries
                        if (width in data.get("pc_train_losses", {}) or 
                            width in data.get("pc_energies", {}) or 
                            width in data.get("bp_losses", {})):
                            has_data = True
                            data_by_n_hidden[n_hidden] = data
                    
                    # Get gamma_0 from first available data
                    if gamma_0 is None and data and data.get("gamma_0") is not None:
                        gamma_0 = data.get("gamma_0")
                
                if not data_by_n_hidden:
                    print(f"  Warning: No data found for width N = {width}")
                    continue
                
                # Get gamma_0: prioritize args.gamma_0 if explicitly provided, otherwise use data value, then fallback
                if gamma_0 is None:
                    gamma_0 = 1.0  # Default fallback
                    print(f"  Warning: gamma_0 not found in data or args, using default {gamma_0}")
                elif args.gamma_0 is not None:
                    print(f"  Using gamma_0 = {gamma_0} from command line argument")
                else:
                    print(f"  Using gamma_0 = {gamma_0} from data directory")
                
                # Create subdirectory for this gamma_0 within the width subdirectory
                gamma_0_plot_dir = os.path.join(plot_dir, f"gamma_0_{gamma_0}")
                os.makedirs(gamma_0_plot_dir, exist_ok=True)
                
                # Print summary of loaded data
                print(f"\nLoaded data summary:")
                print(f"  param_type: {param_type}")
                print(f"  use_skips: {use_skips}")
                print(f"  width N: {width}")
                print(f"  gamma_0: {gamma_0}")
                print(f"  n_hiddens with data: {sorted([n for n in data_by_n_hidden.keys() if n is not None])}")
                # Check for DMFT loss
                has_dmft = any(data.get("dmft_loss") is not None for data in data_by_n_hidden.values())
                print(f"  DMFT loss: {'Yes' if has_dmft else 'No'}")
                print()
                
                # Generate plots (excluding gradient metrics for sp)
                if args.plot == "all" or args.plot == "energies":
                    print("Generating losses and energies plot...")
                    plot_losses_and_energies(data_by_n_hidden, gamma_0_plot_dir, args.colormap, width=width, log_xaxis=args.log_xaxis, n_hiddens=n_hiddens, max_iterations=args.max_iterations)
                    print(f"  Saved to {os.path.join(gamma_0_plot_dir, 'losses_and_energies.pdf')}")
                
                if args.plot == "all" or args.plot == "losses":
                    print("Generating losses plot...")
                    plot_losses(data_by_n_hidden, gamma_0_plot_dir, args.colormap, width=width, log_xaxis=args.log_xaxis, n_hiddens=n_hiddens, max_iterations=args.max_iterations)
                    print(f"  Saved to {os.path.join(gamma_0_plot_dir, 'losses.pdf')}")
                
                # Note: Gradient metrics are excluded for sp param_type
                # (calculate_gradient_metrics and plot_gradient_norms are not called)
            
            print(f"\nCompleted processing for param_type = {param_type}, use_skips = {use_skips}")

    print("\nDone!")
