#!/bin/bash
# set name of job
#SBATCH --job-name=test
# use gpu
#SBATCH --gres=gpu:1  ### Note: Use --gres=gpu:h100:1 for theory
# set the number of nodes
#SBATCH --nodes=1
# set memory per node
#SBATCH --mem=16G     ### Note: Use --mem=32G for alignment analyses
# set max wallclock time
#SBATCH --time=12:00:00
# partition
#SBATCH --partition=short
# select cluster (arc or htc)
#SBATCH --clusters=htc
# qos
#SBATCH --qos=standard
#SBATCH --account=ndcn-computational-neuroscience
# change the location of the .out file
#SBATCH --output=/data/ndcn-computational-neuroscience/mert5001/jpc/experiments/dmft/log/%j.out
### mail alert at start, end and abortion of execution
###SBATCH --mail-type=ALL
### send mail to this address
###SBATCH --mail-user=julian.ngkeekwong@merton.ox.ac.uk

# Specifying virtual envs
module load Anaconda3
source activate $DATA/envs/PC_dmft_env

# run the application
cd $DATA
cd ./jpc/experiments/dmft


######### CONVERGENCE (LINEAR) ##########
#########################################

# # Single (final P x P kernels + sample-traced T x T temporal kernels)
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --plot_temporal_kernels --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_S

# # Across depth (loss curves)
# python analyse_convergence.py --n_samples 20 --n_hiddens 2 3 4 5 --widths 10000 --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_D

# # Across K and gamma (last-layer displacement vs gamma with curves per K; subsumes across K and across gamma above) 
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 20 50 200 500 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --skip_theory --results_dir results_KG

# # Across widths (convergence of kernels + plot final kernels; plots for various depths)
# python analyse_convergence.py --n_samples 20 --n_hiddens 2 3 4 5 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 500 --pc_damping 0.05 --pc_tolerance 1e-10 --n_seeds 5 --results_dir results_W


############ CONVERGENCE (NONLINEAR) ##################
#######################################################

# # Single (final P x P kernels + sample-traced T x T temporal kernels)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --plot_temporal_kernels --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 250 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_S

# # Across gamma (loss curves; includes DMFT) Note: Use H100 (Takes ~20h per gamma otherwise)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 250 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_G

# # Across K and gamma (last-layer displacement vs gamma with curves per K; excludes DMFT) 
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 5 10 20 50 200 500 --n_train_iters 30 --n_fixed_point_steps 250 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --skip_theory --results_dir results_nonlin_KG 

# # Across widths (convergence of kernels + plot final kernels) Note: Use H100 (Takes ~40h otherwise)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 500 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --n_seeds 5 --results_dir results_nonlin_W


############ ALIGNMENT ##################
#########################################

### Iterative inference (tiny-CIFAR10) 

# # Logarithmic loss scale
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align

# # Linear loss scale
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align_L --loss_scale linear


############ BENCHMARKING ##############
########################################

# # MLP, MNIST
# python train_benchmark.py --dataset MNIST --n_epochs 10 --batch_size 64 --width 256 --n_hidden 2 --param_lr 0.3 --param_lr_pc 0.3 --activity_lr 0.01 --n_infer_iters 20 --param_optim adam --act_fn relu --n_seeds 3 --results_dir results_mnist

# # MLP, Fashion-MNIST
# python train_benchmark.py --dataset Fashion-MNIST --n_epochs 10 --batch_size 64 --width 256 --n_hidden 2 --param_lr 0.3 --param_lr_pc 0.3 --activity_lr 0.01 --n_infer_iters 20 --param_optim adam --act_fn relu --n_seeds 3 --results_dir results_fashion_mnist


############ SWEEP ##############
#################################

# # MLP, MNIST: Hyperparameter sweep (Coarse) - Same for Fashion-MNIST (change dataset and name)
# # python train_benchmark.py --dataset Fashion-MNIST --n_epochs 6 --n_seeds 1 \
# python train_benchmark.py --dataset MNIST --n_epochs 6 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 200 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_mnist_sweep_coarse
# #   --results_dir results_fashion_mnist_sweep_coarse

# # MLP, MNIST: Hyperparameter sweep (Fine) - Same for Fashion-MNIST (change dataset and name)
# # python train_benchmark.py --dataset Fashion-MNIST --n_epochs 10 --n_seeds 3 \
# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 3 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_mnist_sweep_fine
# #   --results_dir results_fashion_mnist_sweep_fine
