#!/bin/bash
#SBATCH --time=140:14:14
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
module load cuda/11.1.1
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
cd ~/projects/def-mtaylor3/dikshant/
python maml_cc.py --inner-loops 2 4 6 8 10 12 14 16 --learning-rates 0.001 0.005 0.01 0.015 --train-steps 500
