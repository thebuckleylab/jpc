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
