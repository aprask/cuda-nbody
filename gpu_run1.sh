#!/bin/bash
#SBATCH --job-name=nbod1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
module load cuda
nvidia-smi
nvcc -Xcompiler -fopenmp nbody.cu -lpthread -o nbody
./nbody 1000 10 5 1 gpu