#!/bin/bash
module load python/3.9
module load cuda/11.1.1
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
cd ~/projects/def-mtaylor3/dikshant/
python maml_cc.py --inner-loops [4] --train-steps 10 --learning-rates [0.01]
