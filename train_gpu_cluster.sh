#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=hy5318
#SBATCH --output=/vol/bitbucket/hy5318/erc/logs/train%j.out

source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
export PATH=/vol/bitbucket/hy5318/erc/myvenv/bin/:$PATH
source activate
TERM=vt100
echo "Beginning training job..."
python train-erc-text.py
echo "Finished training job..."
