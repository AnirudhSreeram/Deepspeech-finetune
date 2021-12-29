#!/bin/bash

stage=1

source /data/asreeram/miniconda3/etc/profile.d/conda.sh
conda activate deeppy

if [ $stage -le 1 ]; then

HYDRA_FULL_ERROR=1 python train.py +configs=librispeech #++checkpoint.prefix='librispeec'\
  #++checkpoint.dirpath=/data/asreeram/deepspeech.pytorch/check_point  

fi

