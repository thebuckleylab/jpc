"""Plot learning-regime (γ) sweeps at N=2048 from ``train_toy.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_toy import generate_plots
from train_toy import add_common_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot toy µPC vs BP learning regimes (γ sweep at N=2048)."
        )
    )
    add_common_args(parser)
    parser.set_defaults(
        widths=[2048],
        gamma_0s=[0.1, 0.5, 1.0, 2.0, 4.0],
    )
    return parser.parse_args()


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
