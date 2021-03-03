#!/bin/bash

#SBATCH --job-name=template
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task 10

# change according to your setup
source $HOME/miniconda3/bin/activate std_pt
cd $HOME/cifar-template
python experiment.py