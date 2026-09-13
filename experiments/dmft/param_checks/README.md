# Parameterisation checks

Scripts to reproduce the plots of Figure 1. From this directory 
(`experiments/dmft/param_checks`), run 
```
python train_toy.py \
  --widths 8 16 32 64 128 256 512 1024 2048 \
  --use_skips False
```

Then, to plot selected widths for specific plots, run
```
python plot_toy.py \
  --plot_widths 8 16 32 128 2048
```

For the learning regimes results of Figure 2, run
```
python train_toy.py \
  --widths 2048 \
  --gamma_0s 0.1 0.5 1 2 3 4
python plot_learning_regimes.py
```
