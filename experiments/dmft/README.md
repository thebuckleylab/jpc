# DMFT for PC

## Setup
Clone the `jpc` repo. We recommend using a virtual environment, e.g. 
```
python3 -m venv venv
```
Install `jpc` from the repo root
```
pip install -e .
```
For GPU usage, upgrade jax to the appropriate cuda version (12 as an example 
here).
```
pip install --upgrade "jax[cuda12]==0.5.2"
```
From `experiments/dmft`, install the extra dependencies
```
pip install -r requirements.txt
```

## Reproducing the figures

Figure 1 and Figure A.1-3 are in `param_checks/`. See `param_checks/README.md`.

Figure A.4 is in `saddle-to-saddle/`. See `saddle-to-saddle/README.md`.

Figures 2–3 and the supplementary figures come from the commented blocks in `execute.sh` (convergence, alignment, and benchmarking). Work in `experiments/dmft`.

1. Set `REPLOT=0` at the top of `execute.sh`.
2. Uncomment the block for the figures you want (table below). On the cluster, submit `execute.sh`.
3. Collect the paper figures from those directories:
```
python extract_figures.py
```
This copies them into `figures/pdf` and `figures/png`, and writes a found / missing list to `figures/extract_status.txt`. To redraw the figures from a finished run, run the same `python` command again with `--plot_from_npy` added.

Off the cluster, ignore `execute.sh` (its header is Slurm, and it loads a cluster environment). From `experiments/dmft`, with the environment from Setup, run the uncommented `python` line yourself.

The nonlinear gamma and width runs are slow on a GPU other than an H100. The alignment run needs about 48 GB of memory.

| Figures | Block in `execute.sh` | Directory |
| --- | --- | --- |
| Fig. 2a | Linear: across depth | `results_D` |
| Fig. 2b | Linear: across gamma | `results_G` |
| Fig. 2c, Sup. Fig. 1 | Linear: across widths | `results_W` |
| Fig. 2d–f, Sup. Fig. 2 | Linear: across K and gamma | `results_KG` |
| Fig. 2g–h | Linear: single | `results_S` |
| Fig. 3a | Nonlinear: across gamma | `results_nonlin_G` |
| Fig. 3b, Sup. Fig. 3 | Nonlinear: across K and gamma | `results_nonlin_KG` |
| Fig. 3c | Nonlinear: across widths | `results_nonlin_W` |
| Fig. 3d–i, Sup. Figs. 4–7 | Alignment: iterative inference (`--n_seeds 3`) | `results_alignment` |
| Sup. Fig. 8a | Benchmarking: MLP, MNIST | `results_mnist` |
| Sup. Fig. 8b | Benchmarking: MLP, Fashion-MNIST | `results_fashion_mnist` |
