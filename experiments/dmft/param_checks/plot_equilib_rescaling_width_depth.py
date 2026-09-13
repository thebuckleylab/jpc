"""Plot the inverse equilibrated-energy rescaling as a function of width and depth."""

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import jpc

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    }
)

def parse_bool(value):
    if isinstance(value, bool):
        return value
    lower = str(value).lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean {value!r}; use True/False."
    )


def arch_name(use_skips):
    return "resnet" if use_skips else "mlp"


FIG_SIZE = (10, 8)
FONT_SIZES = {"label": 35, "legend": 25, "tick": 30}
LABEL_PAD = 34
ALPHA = 0.7
DATA_COLOR = "#1E88E5"


def energy_scalings(param_type, gamma, width, depth):
    if param_type == "mupc":
        return gamma**2 * width * depth, float(depth)
    return 1.0, 1.0


def infinite_width_inv_s(widths):
    """Infinite-width init value ``1/s = N`` with ``λ = γ² N L``, ``κ = L``."""
    return np.asarray(widths, dtype=float)


def compute_equilib_rescaling(
    key,
    *,
    input_dim,
    width,
    depth,
    param_type,
    gamma,
    use_skips,
):
    """Closed-form scalar rescaling ``s = S_{00}`` at initialisation."""
    model = jpc.make_mlp(
        key=key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=1,
        act_fn="linear",
        use_bias=False,
        param_type=param_type,
    )
    skip_model = jpc.make_skip_model(depth) if use_skips else None
    x = jr.normal(jr.fold_in(key, 1), (1, input_dim))
    output_energy_scaling, hidden_energy_scaling = energy_scalings(
        param_type, gamma, width, depth
    )
    S = jpc.compute_linear_equilib_rescaling(
        (model, skip_model),
        x,
        param_type=param_type,
        gamma=gamma,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
    )
    return float(jnp.asarray(S)[0, 0])


def sweep_rescalings(
    *,
    widths,
    depths,
    input_dim,
    param_type,
    gamma,
    use_skips,
    seed,
    n_seeds,
):
    widths = list(widths)
    depths = list(depths)
    rescalings = np.zeros((len(widths), len(depths)))
    for seed_offset in range(n_seeds):
        base_key = jr.PRNGKey(seed + seed_offset)
        for i, width in enumerate(widths):
            for j, depth in enumerate(depths):
                key = jr.fold_in(base_key, i * len(depths) + j)
                s = compute_equilib_rescaling(
                    key,
                    input_dim=input_dim,
                    width=width,
                    depth=depth,
                    param_type=param_type,
                    gamma=gamma,
                    use_skips=use_skips,
                )
                rescalings[i, j] += s
                print(
                    f"seed={seed + seed_offset}, N={width}, L={depth}, "
                    f"s={s:.6e}"
                )
    rescalings /= n_seeds
    return rescalings


def plot_rescaling_vs_width_depth(
    widths,
    depths,
    rescalings,
    save_path,
    *,
    param_type,
    linear_z_scale=False,
):
    """3D scatter of ``1/s(θ)`` vs width ``N`` and depth ``L``, plus theory plane."""
    widths = np.asarray(widths)
    depths = np.asarray(depths)
    rescalings = np.asarray(rescalings, dtype=float)

    width_grid, depth_grid = np.meshgrid(widths, depths, indexing="ij")
    widths_flat = width_grid.ravel()
    depths_flat = depth_grid.ravel()
    s_flat = rescalings.ravel()
    finite = np.isfinite(s_flat) & (s_flat != 0)
    if not np.any(finite):
        raise ValueError("All rescaling values are non-finite; nothing to plot.")
    if not np.all(finite):
        n_bad = int(np.size(s_flat) - np.count_nonzero(finite))
        print(f"Warning: dropping {n_bad} non-finite rescaling value(s)")
        widths_flat = widths_flat[finite]
        depths_flat = depths_flat[finite]
        s_flat = s_flat[finite]

    inv_s_flat = 1.0 / s_flat
    log2_widths = np.log2(widths_flat.astype(float))
    log2_depths = np.log2(depths_flat.astype(float))

    if linear_z_scale:
        z_values = inv_s_flat
        offset = 0.0
    else:
        min_inv_s = float(np.min(inv_s_flat))
        offset = 0.0
        inv_s_for_log = inv_s_flat
        if min_inv_s <= 0:
            offset = abs(min_inv_s) + 1e-10
            inv_s_for_log = inv_s_flat + offset
        z_values = np.log10(inv_s_for_log)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        log2_widths,
        log2_depths,
        z_values,
        color=DATA_COLOR,
        s=280,
        alpha=0.9,
        label="Data",
        depthshade=True,
        edgecolors="darkblue",
        linewidths=1,
    )

    plot_theory = param_type != "sp"
    if plot_theory:
        theory_linear = infinite_width_inv_s(widths_flat)
        if not linear_z_scale and offset:
            theory_linear = theory_linear + offset
        theory_z = theory_linear if linear_z_scale else np.log10(theory_linear)
        for i in range(len(log2_widths)):
            ax.plot(
                [log2_widths[i], log2_widths[i]],
                [log2_depths[i], log2_depths[i]],
                [z_values[i], theory_z[i]],
                "k--",
                alpha=0.8,
                linewidth=1,
                zorder=0,
            )

        log2_width_mesh = np.linspace(np.log2(widths.min()), np.log2(widths.max()), 50)
        log2_depth_mesh = np.linspace(np.log2(depths.min()), np.log2(depths.max()), 50)
        W_log2, L_log2 = np.meshgrid(log2_width_mesh, log2_depth_mesh)
        Z_theory_linear = infinite_width_inv_s(2**W_log2)
        if not linear_z_scale and offset:
            Z_theory_linear = Z_theory_linear + offset
        Z_theory = Z_theory_linear if linear_z_scale else np.log10(Z_theory_linear)
        ax.plot_surface(W_log2, L_log2, Z_theory, alpha=0.3, color="gray")

    ax.set_xlabel(r"Width $N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_ylabel(r"Depth $L$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_zlabel(
        r"$1/s(\boldsymbol{\theta})$",
        fontsize=FONT_SIZES["label"],
        labelpad=34,
    )

    width_tick_positions = np.log2(widths.astype(float))
    ax.set_xticks(width_tick_positions)
    ax.set_xticklabels([f"$2^{{{int(p)}}}$" for p in np.round(width_tick_positions)])

    min_depth_log2 = np.log2(float(depths.min()))
    max_depth_log2 = np.log2(float(depths.max()))
    depth_tick_positions = np.arange(
        int(np.ceil(min_depth_log2)), int(np.floor(max_depth_log2)) + 1
    )
    if depth_tick_positions.size == 0:
        depth_tick_positions = np.array([min_depth_log2, max_depth_log2])
        depth_tick_labels = [f"${int(d)}$" for d in depths]
    else:
        depth_tick_labels = [f"$2^{{{p}}}$" for p in depth_tick_positions]
    ax.set_yticks(depth_tick_positions)
    ax.set_yticklabels(depth_tick_labels)

    if linear_z_scale:
        z_tick_positions = np.linspace(float(np.min(inv_s_flat)), float(np.max(inv_s_flat)), 5)
        z_tick_labels = []
        for p in z_tick_positions:
            if abs(p) < 0.01 or abs(p) > 1000:
                z_tick_labels.append(f"${p:.2e}$")
            else:
                z_tick_labels.append(f"${p:.4f}$")
        ax.set_zticks(z_tick_positions)
        ax.set_zticklabels(z_tick_labels)
    else:
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))
        z_tick_positions = np.unique(np.round(np.linspace(z_min, z_max, 3)).astype(int))
        if z_tick_positions.size < 2:
            z_tick_positions = np.linspace(z_min, z_max, 3)
            z_tick_labels = [f"$10^{{{p:.1f}}}$" for p in z_tick_positions]
        else:
            z_tick_labels = [f"$10^{{{p}}}$" for p in z_tick_positions]
        ax.set_zticks(z_tick_positions)
        ax.set_zticklabels(z_tick_labels)

    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis="z", labelsize=FONT_SIZES["tick"], pad=10)
    ax.invert_xaxis()

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DATA_COLOR,
            markersize=10,
            label="Data",
            alpha=ALPHA,
        )
    ]
    if plot_theory:
        handles.append(
            Patch(
                facecolor="gray",
                alpha=0.3,
                label=r"Theory: $\Theta(N)$",
            )
        )
    ax.legend(handles=handles, fontsize=FONT_SIZES["legend"], loc="upper right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.9)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--param_type", type=str, default="mupc", choices=["mupc", "sp"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--use_skips",
        type=parse_bool,
        nargs="*",
        default=None,
        help=(
            "False → MLP, True → residual MLP (resnet). "
            "Omit to run both (default)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 8, 32, 128, 512, 2048],
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
    )
    parser.add_argument("--linear_z_scale", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("results/equilib_energy"),
    )
    args = parser.parse_args()
    use_skips_values = (
        args.use_skips if args.use_skips else [False, True]
    )

    for use_skips in use_skips_values:
        arch = arch_name(use_skips)
        out_dir = args.save_dir / arch
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSweeping {arch} (use_skips={use_skips}) → {out_dir}")
        rescalings = sweep_rescalings(
            widths=args.widths,
            depths=args.depths,
            input_dim=args.input_dim,
            param_type=args.param_type,
            gamma=args.gamma,
            use_skips=use_skips,
            seed=args.seed,
            n_seeds=args.n_seeds,
        )
        np.save(out_dir / "rescalings.npy", rescalings)
        np.save(out_dir / "widths.npy", np.asarray(args.widths))
        np.save(out_dir / "depths.npy", np.asarray(args.depths))
        plot_rescaling_vs_width_depth(
            widths=args.widths,
            depths=args.depths,
            rescalings=rescalings,
            save_path=out_dir / "equilib_rescaling_vs_width_depth.pdf",
            param_type=args.param_type,
            linear_z_scale=args.linear_z_scale,
        )

# MLP only:
# python experiments/dmft/param_checks/plot_equilib_rescaling_width_depth.py --use_skips False
# ResNet (residual MLP) only:
# python experiments/dmft/param_checks/plot_equilib_rescaling_width_depth.py --use_skips True
# Both (default):
# python experiments/dmft/param_checks/plot_equilib_rescaling_width_depth.py
