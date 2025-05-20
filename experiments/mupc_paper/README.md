# μPC paper

<p align='center'>
  <a href='https://thebuckleylab.github.io/jpc/'>
    <img src='.github/mupc_spotlight_fig.svg' />
  </a> 
</p>

This folder contains code to reproduce all the experiments of the paper 
["μPC": Scaling Predictive Coding to 100+ Layer Networks](https://arxiv.org/abs/2505.13124).
For a high-level summary, see [this blog post](https://francesco-innocenti.github.io/posts/2025/05/20/Scaling-Predictive-Coding-to-100+-Layer-Networks/).


## Setup
Clone the `jpc` repo. We recommend using a virtual environment, e.g. 
```
python3 -m venv venv
```
Install `jpc`
```
pip install -e .
```
Now navigate to `mupc_paper` and install all the requirements
```
pip install -r requirements.txt
```
For GPU usage, upgrade jax to the appropriate cuda version (12 as an example 
here).

```
pip install --upgrade "jax[cuda12]"
```

## Compute resources
We recommend using a GPU for the experiments with 64- and 128-layer networks.


## Scripts
* `train_pcn_no_metrics.py`: This is the main script that was used to produce
results for Figs. 1, 5, & A.18-A.19.
* `analyse_activity_hessian`: This script can be used to reproduce results 
related to spectral properties of the activity Hessian at initialisation 
(Figs. 2 & 4, & Figs. A.1-A.7, A.14 & A.24).
* `train_pcn.py`: This was mainly used to monitor the condition number of the
activity Hessian during training (Figs. 3, A.11-A.12, A.15-A.16, A.23, A.25-A.28). 
* `test_energy_theory.py`: This can be used to reproduce results related to
the convergence behaviour of $\mu$PC to BP shown in Section 6 
(Figs. 6 & A.30-A.33).
* `train_bpn.py`: Used to obtain all the results with backprop. 
* `test_mlp_fwd_pass.py`: Used for results of Fig. A.13.
* `toy_experiments.py`: Used for many secondary results in the Appendix. 

The majority of results are plotted in `plot_results.ipynb` under informative
headings. For details of all the experiments, see Section A.4 of the paper.
