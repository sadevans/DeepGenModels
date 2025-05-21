CODE_DIR=./

GPUS=4
N_PROC=1
GPU=4

for EXP in  1

do
echo gan$EXP
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1252 --nproc_per_node $N_PROC $CODE_DIR/code/train.py --config-dir $CODE_DIR/configs --config-name gan$EXP > log 2> mistakes
done