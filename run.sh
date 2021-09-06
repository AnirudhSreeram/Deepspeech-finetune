#!/bin/bash
stage=1


source /home/asreeram/miniconda3/etc/profile.d/conda.sh

conda activate deeppy

if [ $stage -le 1 ]; then

HYDRA_FULL_ERROR=1 python train.py +configs=librispeech trainer.gpus=1 \
#	trainer.resume_from_checkpoint=True

fi

