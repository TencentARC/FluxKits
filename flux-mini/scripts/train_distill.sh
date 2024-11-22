export FLUX_DEV=/path/to/flux-dev
export FLUX_MINI=/path/to/flux-mini
export AE=/path/to/flux-ae


HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=8

torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=14539 --node_rank=$INDEX \
 train_flux_distill.py --config "train_configs/distill.yaml"