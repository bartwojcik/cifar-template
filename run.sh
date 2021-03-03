#!/bin/bash

#SBATCH --job-name=cifar-template
#SBATCH --qos=quick
#SBATCH --gres=gpu:1

# change to your environment name here
source $HOME/miniconda3/bin/activate std_pt

cd $HOME/cifar-template
python experiment.py