CODE_DIR=./

GPUS=5
N_PROC=1
GPU=5

for EXP in  5

do
echo gan$EXP
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1253 --nproc_per_node $N_PROC $CODE_DIR/code/train.py --config-dir $CODE_DIR/configs --config-name gan$EXP > log 2> mistakes
done