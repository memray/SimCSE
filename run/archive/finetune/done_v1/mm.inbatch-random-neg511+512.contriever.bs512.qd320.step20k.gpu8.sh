#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_RANK=0
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=4
export MAX_STEPS=20000
export BEIR_DATASETS='nfcorpus fiqa arguana scidocs scifact webis-touche2020 cqadupstack trec-covid'

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export EXP_NAME=mm.inbatch-random-neg511+512.arch-inbatch.model-contriever.avg.dot.qd320.step20k.bs512.lr1e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/finetune/$EXP_NAME
mkdir -p $PROJECT_DIR
rm -rf $PROJECT_DIR/*
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export WANDB_PROJECT=unsup_retrieval_finetune
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb

sleep 3h
cd /export/share/ruimeng/project/search/uir_best_cc
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=31133 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --reload_model_from facebook/contriever --finetune true --arch_type inbatch --queue_size 0 --train_file /export/home/data/search/msmarco/msmarco-bm25.jsonl  --neg_indices 3 --negative_strategy random --dev_file /export/home/data/pretrain/wiki2021_structure/wiki_psgs_w100.dev.tail2e13.tsv --data_type hf --remove_unused_columns False --sim_type dot --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 1000 --logging_steps 100 --eval_steps 5000 --save_steps $MAX_STEPS --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --beir_batch_size 256 --beir_datasets $BEIR_DATASETS --dataloader_num_workers $NUM_WORKER --learning_rate 1e-5 --max_seq_length 320 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 577 --report_to wandb > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid

