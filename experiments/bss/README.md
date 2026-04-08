# Predictive dendrites for blind source separation


## Setup
Clone the `jpc` repo. We recommend using a virtual environment, e.g. 
```
python3 -m venv venv
```
Install `jpc`
```
pip install -e .
```
Now navigate to `experiments/bss` and install all the requirements
```
pip install -r requirements.txt
```


## Scripts
Run the main training script:
```
python train_bss.py
```
and plot the results with
```
python plot_bss.py
```
