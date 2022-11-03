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

cd /export/share/ruimeng/project/search/uir_best_cc
nohup python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8089 mine_negatives.py --model_name_or_path /export/home/exp/search/unsup_dr/exp_v3/v2code+prompt_in_beir.cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs2048.lr3e5.gpu16/checkpoints/checkpoint-200000 --metric dot --per_gpu_batch_size 1024 --beir_data_path /export/home/data/beir/ --output_dir /export/home/exp/search/unsup_dr/exp_v3/v2code+prompt_in_beir.cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs2048.lr3e5.gpu16/checkpoints/checkpoint-200000/msmarco_epoch0 --add_qd_prompt true --use_faiss true >/export/home/exp/search/unsup_dr/exp_v3/v2code+prompt_in_beir.cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs2048.lr3e5.gpu16/checkpoints/checkpoint-200000/mine_negatives-msmarco-epoch0.log 2>&1 & echo $! > run.pid
echo "/export/home/exp/search/unsup_dr/exp_v3/v2code+prompt_in_beir.cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs2048.lr3e5.gpu16/checkpoints/checkpoint-200000/mine_negatives-msmarco-epoch0.log"


