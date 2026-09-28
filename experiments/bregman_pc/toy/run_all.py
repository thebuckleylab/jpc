"""Regenerate all toy intuition figures for linear, tanh, and sigmoid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_bregman_geometry import plot_all as plot_bregman_geometry
from plot_matched_potential import (
    ACT_FNS,
    _DEFAULT_SAVE,
    _SAVE_DIR_HELP,
    plot_all as plot_matched_potential,
)
from plot_mirror_flow import plot_all as plot_mirror_flow


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run every toy plot script for each activation."
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=str(_DEFAULT_SAVE),
        help=_SAVE_DIR_HELP,
    )
    p.add_argument(
        "--act-fn",
        nargs="*",
        default=list(ACT_FNS),
        choices=ACT_FNS,
        help="Activations to plot (default: linear tanh sigmoid).",
    )
    args = p.parse_args()
    for name in args.act_fn:
        print(f"=== {name} ===")
        plot_matched_potential(name, args.save_dir)
        plot_bregman_geometry(name, args.save_dir)
        plot_mirror_flow(name, args.save_dir)


if __name__ == "__main__":
    main()
