#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_RANK=0
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=4
export MAX_STEPS=100000

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export EXP_NAME=wikipsg.seed477.moco-2e17.contriever256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/wikipsg_v1/$EXP_NAME
mkdir -p $PROJECT_DIR
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export WANDB_PROJECT=unsup_retrieval_wikipsg
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb

cd /export/home/project/search/uir_best_cc
#nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=3112 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type moco --train_file /export/home/data/search/wiki/wiki_phrase.jsonl --dev_file /export/home/data/pretrain/wiki2021_structure/wiki_psgs_w100.dev.tail2e13.tsv --data_type hf --data_pipeline_name contriever256 --remove_unused_columns False --sim_type dot --queue_size 131072 --momentum 0.9995 --output_dir $PROJECT_DIR --overwrite_output_dir --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 10000 --logging_steps 100 --eval_steps 10000 --save_steps 10000 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --beir_batch_size 256 --dataloader_num_workers $NUM_WORKER --learning_rate 5e-5 --max_seq_length 128 --evaluation_strategy steps --load_best_model_at_end --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 477 --report_to wandb --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid

export WANDB_RUN_ID=36i4wv57
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=3112 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type inbatch --reload_model_from $PROJECT_DIR --remove_unused_columns False --sim_type dot --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --do_eval --run_name $EXP_NAME --fp16 --seed 477 --report_to wandb --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup-eval.log 2>&1 & echo $! > run.pid

