# DMFT for PC

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


# TODOs (ordered by priority)
* Determine whether PC -> BP at large width in the rich regime ($\gamma=1$).
* Check $h^*$ and deviation from $\hat{h}$.
* Extend `test_equilib_energy.py` to training (including simple classification tasks).
* Check empirics for L>2.
* Compute DMFT correlation and response functions.
