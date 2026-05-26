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
We recommend using a GPU for the experiments in Figures 1 & 4, detailed below.


## Scripts
Below are specific argument configurations that can be used to reproduce the 
main experiments in the paper. For details of all the experiments, see Section 
A.8 of the paper.
* **Nonlinear networks (Figure 4)**: For the results with residual MLPs, in `/limits_paper` run
  ```bash
  python train.py \
    --dataset CIFAR10 \
    --n_samples 64 \
    --act_fn tanh \
    --param_types mupc \
    --use_skips True \
    --param_optim adam \
    --param_lr 1e-3 \
    --infer_mode optim \
    --n_infer_iters 100000 \
    --activity_lrs 0.3 \
    --n_hiddens 31 \
    --widths 8 16 32 64 128 256 512 \
    --n_seeds 1
  ```
  For the CNN results, in `/limits_paper/cnn` run
  ```bash
  python test_theory.py \
    --dataset imagenet \
    --widths 2 8 16 32 \
    --n_res_blocks 3 \
    --param_type mupc \
    --act_fn tanh \
    --additive_depth_factor 4 \
    --batch_size 128 \
    --param_optim adam \
    --param_lr 1e-3 \
    --loss_id ce \
    --n_infer_iters 200 \
    --activity_lrs 0.3
  ```
  Finally, for the transformer results, in `/limits_paper/transformer` run
  ```bash
  python train.py \
    --seq_len 32 \
    --batch_size 128
    --d_models 8 16 32 64 128 256 512 \
    --n_blocks 12 \
    --n_heads 8 \
    --param_type mupc \
    --use_layer_norm False \
    --use_softmax True \
    --act_fn gelu \
    --init_std 0.02 \
    --param_lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.95 \
    --adam_eps 1e-12 \
    --weight_decay 0. \
    --n_infer_iters 800 \
    --activity_lrs 0.45
  ```
* **Figure 1**: In `/limits_paper` run
  ```bash
  python train.py \
    --dataset CIFAR10 \
    --n_samples 64 \
    --act_fn linear \
    --param_types mupc \
    --use_skips True \
    --param_optim adam \
    --param_lr 1e-3 \
    --infer_mode closed_form \
    --n_hiddens 1 3 7 15 31 \
    --widths 2 8 32 128 512 2048 \
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
