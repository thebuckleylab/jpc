#!/bin/bash
# set name of job
#SBATCH --job-name=test
# use gpu
#SBATCH --gres=gpu:1
# set the number of nodes
#SBATCH --nodes=1
# set memory per node
#SBATCH --mem=16G
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

# Single (final P x P kernels + sample-traced T x T kernels)
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --plot_temporal_kernels --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_S

# Across depth
# python analyse_convergence.py --n_samples 20 --n_hiddens 2 3 4 5  --widths 10000 --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_D

# Across gamma
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_G

# Across K (DMFT only for smallest K; stacked kernel grid + displacement)
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 20 50 200 500 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_K

# Across K and gamma (last-layer displacement vs gamma, curves per K)
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 20 50 200 500 --n_train_iters 20 --n_fixed_point_steps 100 --pc_damping 0.05 --results_dir results_KG

# Across widths (convergence of kernels + plot final kernels)
# python analyse_convergence.py --n_samples 20 --n_hiddens 5 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 20 --n_fixed_point_steps 500 --pc_damping 0.05 --pc_tolerance 1e-10 --n_seeds 3 --results_dir results_W


############ CONVERGENCE (NONLINEAR) ##################
#######################################################

# Single (final P x P kernels + sample-traced T x T kernels)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --plot_temporal_kernels --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_S --skip_theory

# Across depth
# python analyse_convergence.py --n_samples 8 --n_hiddens 1 2 3  --widths 10000 --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_D --skip_theory

# Across gamma
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_G --skip_theory

# Across K (DMFT only for smallest K; stacked kernel grid + displacement)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 50 100 500 1000 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_K --skip_theory

# Across K and gamma (last-layer displacement vs gamma, curves per K)
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10000 --gamma_0s 0.1 0.5 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 50 100 500 1000 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_nonlin_KG --skip_theory

# Across widths (convergence of kernels + plot final kernels)
## python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --n_seeds 3 --results_dir results_nonlin_W --skip_theory



######### CHECK CONVERGENCE ##########
######################################

# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 100 --pc_damping 0.1 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results0
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 200 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results1
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 200 --pc_damping 0.1 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results2
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 300 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results3
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 400 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results4
# python analyse_convergence.py --n_samples 8 --n_hiddens 3 --widths 10 25 100 250 1000 2500 10000 --plot_mode both --gamma_0s 1.0 --param_lr_pc 1.0 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 30 --n_fixed_point_steps 500 --pc_damping 0.05 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results5


############ ALIGNMENT ##################
#########################################

# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align1
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align2
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.4 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align3
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.4 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align4

# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align1L --loss_scale linear
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align2L --loss_scale linear
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.4 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align3L --loss_scale linear
# python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.4 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 500 --n_train_iters 1001 --act_fn tanh --dataset tiny-CIFAR10 --results_dir results_align4L --loss_scale linear



############ SWEEP ##############
#################################

# python train_benchmark.py --dataset MNIST --n_epochs 6 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 50 500 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_bs64


# python train_benchmark.py --dataset MNIST --n_epochs 3 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 16 \
#   --param_lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
#   --param_lr_pc 0.01 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 50 500 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_bs16


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 2 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_bs64_fine


# python train_benchmark.py --dataset MNIST --n_epochs 5 --n_seeds 2 \
#   --width 256 --n_hidden 2 --batch_size 16 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.01 0.03 0.1 0.3 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_bs16_fine



###### Check activity_lr #########

# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 2 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_adam_relu_2


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 2 \
#   --param_optim sgd_momentum --act_fn relu \
#   --results_dir results_sweep_sgdm_relu_2


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 2 \
#   --param_optim gd --act_fn relu \
#   --results_dir results_sweep_gd_relu_2


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_sweep_adam_relu_20


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 20 \
#   --param_optim sgd_momentum --act_fn relu \
#   --results_dir results_sweep_sgdm_relu_20


# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 0.03 0.1 0.3 1.0 3.0 \
#   --param_lr_pc 0.03 0.1 0.3 1.0 3.0 \
#   --activity_lr 0.00001 0.0001 0.001 0.01 0.1 \
#   --n_infer_iters 20 \
#   --param_optim gd --act_fn relu \
#   --results_dir results_sweep_gd_relu_20


### Single test ###

# python train_benchmark.py --dataset MNIST --n_epochs 10 --n_seeds 1 \
#   --width 256 --n_hidden 2 --batch_size 64 \
#   --param_lr 1.0 \
#   --param_lr_pc 1.0 \
#   --activity_lr 0.001 \
#   --n_infer_iters 20 \
#   --param_optim adam --act_fn relu \
#   --results_dir results_test --skip_bp