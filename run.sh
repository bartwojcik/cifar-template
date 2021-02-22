#!/bin/bash

#SBATCH --job-name=cifar-template
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

source $HOME/miniconda3/bin/activate std_pt

cd $HOME/cifar-template
python experiment.py