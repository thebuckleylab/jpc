"""Euclidean vs mirror inference on a tiny deep Bregman-PC MLP.

    F(z1, z2) = D_psi(z1, phi(w1 x)) + D_psi(z2, phi(w2 z1))
                + (1/2)(w3 z2 - y)^2

Figures: primal/dual trajectories (``mirror_flow_primal_dual.pdf``) and
energy vs inference time (``mirror_flow_energy.pdf``).

PDFs go to ``<save-dir>/<act-fn>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_matched_potential import (
    ALPHA,
    BREGMAN_COLOR,
    FIG_SIZE,
    FONT_SIZES,
    LINE_WIDTH,
    STD_PC_COLOR,
    _clip_activity,
    _style_axes,
    act_save_dir,
    add_act_save_args,
    bregman,
    inv_phi,
    phi,
    save_fig,
)

DT = 0.04
N_STEPS = 300
N_CONTOUR = 12
MARK_EVERY = 22
ARROW_FRAC = 0.04
STAR_SKIP_FRAC = 0.04
ENERGY_REL_THRESH = 0.03
ENERGY_START_IDX = (0, 2, 6)
DUAL_HALF = 1.5
START_LINESTYLES = ("-", "--", "-.")


def _primal_lim(name: str) -> tuple[float, float]:
    if name == "tanh":
        return -1.0, 1.0
    if name == "sigmoid":
        return 0.0, 1.0
    return -3.5, 3.5


def _interior_grid(lo: float, hi: float, name: str, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    pad = 0.03 * (hi - lo) if name != "linear" else 0.0
    g = np.linspace(lo + pad, hi - pad, n)
    return np.meshgrid(g, g)


def _rk4(f, x0: np.ndarray, dt: float, n_steps: int, clip=None) -> np.ndarray:
    x = np.asarray(x0, dtype=float)
    traj = np.empty((n_steps + 1, x.size))
    traj[0] = x
    for i in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if clip is not None:
            x = clip(x)
        traj[i + 1] = x
    return traj


def _style_flow_ax(ax, xlabel, ylabel, title, fonts):
    ax.set_title(title, fontsize=fonts["title"], pad=14)
    _style_axes(ax, xlabel, ylabel, fonts)


def _method_handles():
    return [
        Line2D([0], [0], color=STD_PC_COLOR, lw=LINE_WIDTH, label="Euclidean"),
        Line2D([0], [0], color=BREGMAN_COLOR, lw=LINE_WIDTH, label="Mirror"),
    ]


def _legend_handles():
    return [
        *_method_handles(),
        Line2D([0], [0], marker="o", color="black", lw=0, label="Start"),
        Line2D([0], [0], marker="*", color="black", lw=0, markersize=14, label="Equilibrium"),
    ]


def _pair_starts(name: str) -> np.ndarray:
    if name == "tanh":
        return np.array([
            [-0.88, -0.82],
            [-0.88, 0.12],
            [-0.85, 0.80],
            [0.10, -0.88],
            [0.82, -0.85],
            [0.88, 0.12],
            [0.82, 0.82],
        ])
    if name == "sigmoid":
        return np.array([
            [0.12, 0.12],
            [0.12, 0.50],
            [0.12, 0.88],
            [0.50, 0.12],
            [0.88, 0.12],
            [0.88, 0.50],
            [0.88, 0.88],
        ])
    return np.array([
        [-2.6, -2.4],
        [-2.6, 0.2],
        [-2.6, 2.4],
        [0.2, -2.6],
        [2.4, -2.4],
        [2.6, 0.2],
        [2.4, 2.4],
    ])


def _net() -> dict:
    x = 0.45
    w1, w2, w3, y = 0.90, -0.70, 0.85, 0.55
    return {"x": x, "w1": w1, "w2": w2, "w3": w3, "y": y}


def _energy(z: np.ndarray, net: dict, name: str) -> float:
    z1, z2 = float(z[0]), float(z[1])
    a1 = net["w1"] * net["x"]
    a2 = net["w2"] * z1
    return float(
        bregman(name, z1, phi(name, a1))
        + bregman(name, z2, phi(name, a2))
        + 0.5 * (net["w3"] * z2 - net["y"]) ** 2
    )


def _grad(z: np.ndarray, net: dict, name: str) -> np.ndarray:
    zc = _clip_activity(name, z)
    z1, z2 = float(zc[0]), float(zc[1])
    a1 = net["w1"] * net["x"]
    a2 = net["w2"] * z1
    zhat1 = phi(name, a1)
    zhat2 = phi(name, a2)
    g1 = float(inv_phi(name, z1) - inv_phi(name, zhat1))
    g1 += -net["w2"] * (z2 - zhat2)
    g2 = float(inv_phi(name, z2) - inv_phi(name, zhat2))
    g2 += net["w3"] * (net["w3"] * z2 - net["y"])
    return np.array([g1, g2])


def _F_mesh(Z1, Z2, F_fn) -> np.ndarray:
    out = np.empty(Z1.shape)
    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            out[i, j] = F_fn(np.array([Z1[i, j], Z2[i, j]]))
    return out


def _euclidean_traj(z0, grad_fn, name):
    def f(z):
        return -grad_fn(_clip_activity(name, z))

    return _rk4(f, z0, DT, N_STEPS, clip=lambda z: _clip_activity(name, z))


def _mirror_traj(z0, grad_fn, name):
    u0 = inv_phi(name, z0)

    def f(u):
        z = phi(name, u)
        return -grad_fn(z)

    u_traj = _rk4(f, u0, DT, N_STEPS)
    return np.stack([phi(name, u_traj[:, 0]), phi(name, u_traj[:, 1])], axis=1)


def _equilibrium(grad_fn, name) -> np.ndarray:
    z0 = np.array([0.1, 0.1]) if name != "sigmoid" else np.array([0.5, 0.5])
    return _mirror_traj(z0, grad_fn, name)[-1]


def _contour_levels(Z) -> np.ndarray | None:
    finite = np.isfinite(Z)
    if not np.any(finite):
        return None
    lo, hi = np.quantile(Z[finite], 0.02), np.quantile(Z[finite], 0.92)
    if hi <= lo:
        return None
    return np.linspace(lo, hi, N_CONTOUR)


def _draw_contours(ax, X, Y, Z, z_star_xy, levels=None):
    if levels is None:
        levels = _contour_levels(Z)
    if levels is not None:
        ax.contour(X, Y, Z, levels=levels, colors="0.75", linewidths=1.0)
    ax.plot(*z_star_xy, marker="*", color="black", markersize=16, zorder=6)


def _time_idx(traj: np.ndarray) -> np.ndarray:
    return np.arange(MARK_EVERY, len(traj) - 1, MARK_EVERY)


def _to_dual(z_traj: np.ndarray, name: str) -> np.ndarray:
    return np.stack([inv_phi(name, z_traj[:, 0]), inv_phi(name, z_traj[:, 1])], axis=1)


def _draw_speed_arrows(ax, traj, color, star, span: float, width: float = 0.009):
    idx = _time_idx(traj)
    if idx.size == 0:
        return
    pts = traj[idx]
    d = traj[idx + 1] - pts
    n = np.hypot(d[:, 0], d[:, 1])
    keep = n > 1e-4
    keep &= np.hypot(pts[:, 0] - star[0], pts[:, 1] - star[1]) > STAR_SKIP_FRAC * span
    pts, d, n = pts[keep], d[keep], n[keep]
    if pts.size == 0:
        return
    arrow_len = ARROW_FRAC * span
    U = arrow_len * d[:, 0] / n
    V = arrow_len * d[:, 1] / n
    ax.quiver(
        pts[:, 0], pts[:, 1], U, V, color=color, angles="xy", scale_units="xy", scale=1.0,
        width=width, headwidth=4.2, headlength=5.0, headaxislength=4.4,
        minlength=0.0, pivot="mid", zorder=5,
    )


def _draw_pairs(ax, euc, mir, starts, z_star, span: float):
    for e_traj, m_traj in zip(euc, mir):
        ax.plot(e_traj[:, 0], e_traj[:, 1], color=STD_PC_COLOR, lw=LINE_WIDTH, ls="-", alpha=ALPHA)
        ax.plot(m_traj[:, 0], m_traj[:, 1], color=BREGMAN_COLOR, lw=LINE_WIDTH, ls="-", alpha=ALPHA)
        _draw_speed_arrows(ax, e_traj, STD_PC_COLOR, z_star, span)
        _draw_speed_arrows(ax, m_traj, BREGMAN_COLOR, z_star, span)
    ax.scatter(starts[:, 0], starts[:, 1], s=36, color="black", zorder=6)


def _edge_ticks(ax, lo: float, hi: float) -> None:
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"], pad=6)
    ticks = [lo, 0.5 * (lo + hi), hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def _plot_primal_dual(name, save_dir, F_fn, euc, mir, starts, z_star, xlab, ylab, filename):
    z_lo, z_hi = _primal_lim(name)
    Z1, Z2 = _interior_grid(z_lo, z_hi, name)
    Fz = _F_mesh(Z1, Z2, F_fn)
    u_lo, u_hi = -DUAL_HALF, DUAL_HALF
    U1, U2 = _interior_grid(u_lo, u_hi, "linear")
    Fu = _F_mesh(U1, U2, lambda u: F_fn(np.array([phi(name, u[0]), phi(name, u[1])])))
    levels = _contour_levels(Fz)
    u_star = np.array([inv_phi(name, z_star[0]), inv_phi(name, z_star[1])])
    u_starts = _to_dual(starts, name)
    euc_u = [_to_dual(traj, name) for traj in euc]
    mir_u = [_to_dual(traj, name) for traj in mir]
    fonts = FONT_SIZES

    fig_w, fig_h = 15.2, 5.6
    side = 3.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    wf, hf = side / fig_w, side / fig_h
    y0 = 0.20
    x0 = 0.10
    gap = 0.11
    axes = [
        fig.add_axes([x0, y0, wf, hf]),
        fig.add_axes([x0 + wf + gap, y0, wf, hf]),
    ] 

    _draw_contours(axes[0], Z1, Z2, Fz, z_star, levels=levels)
    _draw_pairs(axes[0], euc, mir, starts, z_star, z_hi - z_lo)
    axes[0].set_xlim(z_lo, z_hi)
    axes[0].set_ylim(z_lo, z_hi)
    _style_flow_ax(axes[0], xlab, ylab, r"(a) Primal", fonts)

    _draw_contours(axes[1], U1, U2, Fu, u_star, levels=levels)
    _draw_pairs(axes[1], euc_u, mir_u, u_starts, u_star, u_hi - u_lo)
    axes[1].set_xlim(u_lo, u_hi)
    axes[1].set_ylim(u_lo, u_hi)
    _style_flow_ax(axes[1], r"$u^1$", r"$u^2$", r"(b) Dual", fonts)

    _edge_ticks(axes[0], z_lo, z_hi)
    _edge_ticks(axes[1], u_lo, u_hi)
    axes[0].set_position([x0, y0, wf, hf])
    axes[1].set_position([x0 + wf + gap, y0, wf, hf])

    fig.legend(
        handles=_legend_handles(),
        fontsize=fonts["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(x0 + 2 * wf + gap + 0.02, y0 + hf),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    path = os.path.join(save_dir, filename)
    save_fig(fig, path, tight=False)


def _plot_energy(save_dir, F_fn, euc, mir, filename):
    tau = np.arange(N_STEPS + 1) * DT
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    curves = []
    for e_traj, m_traj, ls in zip(euc, mir, START_LINESTYLES):
        f_euc = np.array([F_fn(z) for z in e_traj])
        f_mir = np.array([F_fn(z) for z in m_traj])
        curves.extend((f_euc, f_mir))
        ax.plot(tau[1:], f_euc[1:], color=STD_PC_COLOR, lw=LINE_WIDTH, ls=ls, alpha=ALPHA)
        ax.plot(tau[1:], f_mir[1:], color=BREGMAN_COLOR, lw=LINE_WIDTH, ls=ls, alpha=ALPHA)
    stacked = np.stack(curves)
    ymax = float(stacked[:, 0].max())
    ymin = float(np.nanmin(stacked[stacked > 0]))
    settled = np.all(stacked <= ENERGY_REL_THRESH * ymax, axis=0)
    tau_max = float(tau[int(np.argmax(settled))]) if np.any(settled) else float(tau[-1])
    _style_axes(ax, r"$\tau$", "Energy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(DT, max(tau_max, DT * 2))
    ax.set_ylim(0.8 * ymin, 1.2 * ymax)
    ax.legend(
        handles=_method_handles(),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        loc="upper right",
    )
    save_fig(fig, os.path.join(save_dir, filename))


def plot_all(name: str, save_dir: str | os.PathLike) -> None:
    save_dir = act_save_dir(save_dir, name)
    net = _net()
    F_fn = lambda z: _energy(z, net, name)
    grad_fn = lambda z: _grad(z, net, name)
    starts = _pair_starts(name)
    z_star = _equilibrium(grad_fn, name)
    euc = [_euclidean_traj(z0, grad_fn, name) for z0 in starts]
    mir = [_mirror_traj(z0, grad_fn, name) for z0 in starts]
    energy_idx = list(ENERGY_START_IDX)
    _plot_primal_dual(
        name, save_dir, F_fn, euc, mir, starts, z_star,
        r"$z^1$", r"$z^2$", "mirror_flow_primal_dual.pdf",
    )
    _plot_energy(
        save_dir, F_fn,
        [euc[i] for i in energy_idx],
        [mir[i] for i in energy_idx],
        "mirror_flow_energy.pdf",
    )


if __name__ == "__main__":
    args = add_act_save_args(
        argparse.ArgumentParser(
            description="Paired Euclidean/mirror trajectories on a tiny Bregman-PC energy."
        )
    ).parse_args()
    plot_all(args.act_fn, args.save_dir)
