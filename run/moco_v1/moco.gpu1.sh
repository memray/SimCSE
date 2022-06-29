#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=4
export LOCAL_RANK=0
export WORLD_SIZE=1

export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=0
export MAX_STEPS=200000

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export EXP_NAME=wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.gpu1.bs64.step200k.warmup10k.lr1e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/exp_v1/$EXP_NAME
mkdir -p $PROJECT_DIR
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export WANDB_PROJECT=unsup_retrieval_v1
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb


cd /export/share/ruimeng/project/search/simcse
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=22799 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type moco --train_file wikipedia --data_type hf --data_pipeline_name contriever-256 --remove_unused_columns False --sim_type dot --queue_size 4096 --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 10000 --logging_steps 100 --eval_steps 5000 --save_steps 5000 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --dataloader_num_workers $NUM_WORKER --preprocessing_num_workers $NUM_WORKER --learning_rate 1e-5 --max_seq_length 256 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --fp16 --run_name $EXP_NAME > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid
