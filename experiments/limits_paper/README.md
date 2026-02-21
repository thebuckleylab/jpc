# On the ♾️ Width & Depth Limits of PCNs

[![Paper](https://img.shields.io/badge/Paper-arXiv:2508.01191-%23f2806bff.svg)](https://www.arxiv.org/abs/2602.07697)

Code for the paper [On the Infinite Width and Depth Limits of Predictive Coding Networks](https://www.arxiv.org/abs/2602.07697). 


## Setup
Clone the `jpc` repo. We recommend using a virtual environment, e.g. 
```
python3 -m venv venv
```
Install `jpc`
```
pip install -e .
```
For GPU usage, upgrade jax to the appropriate cuda version (12 as an example 
here).

```
pip install --upgrade "jax[cuda12]==0.5.2"
```
Now navigate to `experiments/limits_paper` and install all the requirements

```
pip install -r requirements.txt
```


## Compute resources
We recommend using a GPU for the **Image classification tasks** below.


## Scripts
All experiments rely on the `train.py` script. Below are the specific argument configurations that can be used to reproduce the main experiments in the paper. 
For details of all the experiments, see Section A.8 of the paper.
* **Image classification tasks**: for the results of Figure 1, in `/limits_paper` run
  ```bash
  python train.py \
    --dataset CIFAR10 \
    --n_samples 64 \
    --use_skips True \
    --param_optim adam \
    --param_lr 1e-3 \
    --n_hiddens 1 3 7 15 31 \
    --widths 2 8 32 128 512 2048 \
    --n_seeds 3
  ```
  For the results of Figure 4, run 
  ```bash
  python train.py \
    --dataset CIFAR10 \
    --n_samples 64 \
    --act_fn tanh \
    --use_skips True \
    --param_optim adam \
    --param_lr 1e-3 \
    --infer_mode optim \
    --activity_lrs 0.1 0.5 1 5 10 20
    --n_hiddens 1 15 \
    --widths 2048 \
    --n_seeds 3
  ```
* **Toy task**: run `python train.py --param_types mupc` for results of Figures 2 & 
A.17 (w/ and w/o skips, respectively). Run `python train.py --param_types sp`
for comparative results of Figures A.18-19 with the standard parameterisation.
* **BP learning regimes**: run `python train.py --widths 2048 --gamma_0s 0.1 0.5 1 2 3 4` 
for results of Figures A.3 and A.22 (w/ and w/o skips).
* **Saddle-to-saddle PC regime**: run `python train.py --n_train_iters 1000 --n_hiddens 1 3 5 7 --widths 1 4 8` for results of Figures A.4 and A.23 (w/ and w/o skips).

Results are plotted with self-explanatory `plot_{}.py` scripts, which were generated with the assistance of Cursor. For example, `plot_width_vs_depth_results.py` plots results like those in Figure 1.


## Citation
If you find this work useful, please cite:

```bibtex
@article{innocenti2026infinite,
  title={On the Infinite Width and Depth Limits of Predictive Coding Networks},
  author={Innocenti, Francesco and Achour, El Mehdi and Bogacz, Rafal},
  journal={arXiv preprint arXiv:2602.07697},
  year={2026}
}
```
Also consider starring the repo! ⭐️
