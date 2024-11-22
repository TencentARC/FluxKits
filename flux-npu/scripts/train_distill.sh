export FLUX_DEV=/path/to/flux-dev
export FLUX_MINI=/path/to/flux-mini
export AE=/path/to/flux-ae

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
export UCX_NET_DEVICES=eth1
export NCCL_IB_HCA=mlx5_eth_1,mlx5_eth_5,mlx5_eth_3,mlx5_eth_7,mlx5_eth_4,mlx5_eth_8,mlx5_eth_2,mlx5_eth_6
export GLOO_SOCKET_IFNAME=eth1

source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh


HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=16

torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=14539 --node_rank=$INDEX \
 train_flux_distill.py --config "train_configs/distill.yaml"