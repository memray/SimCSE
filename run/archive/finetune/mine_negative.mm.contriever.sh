#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TRANSFORMERS_CACHE=/export/home/cache/hf

CKPT_PATH="/export/home/exp/search/unsup_dr/finetune/mm.inbatch-random-neg1023+1024.arch-inbatch.model-contriever.avg.dot.qd192.step20k.bs1024.lr1e5/checkpoints/checkpoint-20000/"
mkdir -p ${CKPT_PATH}/msmarco_minedneg

cd /export/share/ruimeng/project/search/uir_best_cc
nohup python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8089 mine_negatives.py --model_name_or_path ${CKPT_PATH} --per_gpu_batch_size 512 --beir_data_path /export/home/data/beir/ --output_dir ${CKPT_PATH}/msmarco_minedneg --use_faiss true > ${CKPT_PATH}/msmarco_minedneg/mine_negatives.log 2>&1 & echo $! > run.pid
echo "${CKPT_PATH}/msmarco_minedneg/mine_negatives.log"
