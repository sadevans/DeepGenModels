#!/bin/bash

CODE_DIR=/netapp/a.gorokhova/itmo/DeepGenModels/hw1

GPUS=7
N_PROC=1
GPU=7

for EXP in _mnad6
do
    echo anomaly$EXP
    export HYDRA_FULL_ERROR=1
    CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1254 --nproc_per_node $N_PROC $CODE_DIR/code/infer.py --config-dir $CODE_DIR/configs --config-name anomaly$EXP
done