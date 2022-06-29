#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_RANK=0
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=true
export NUM_WORKER=16
export MAX_LEN=512
export MAX_STEPS=100000

export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=simcse_v2
export NCCL_DEBUG=INFO
export WANDB_DIR=/export/home/exp/search/simcse/wandb_logs

export EXP_NAME=psg64.wiki-docs.bert-base-uncased.dual.avg-mlp.cosine.step100k.bs256.lr1e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/simcse_v2/$EXP_NAME
export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
mkdir -p $PROJECT_DIR
echo $PROJECT_DIR


cd /export/share/ruimeng/project/search/simcse
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=11788 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --shared_encoder False --hidden_dropout_prob 0.1 --train_file /export/home/data/pretrain/wiki2021_structure/wiki_docs.jsonl --data_type document --data_pipeline_name psg-64 --cl_loss_weights [1.0,0.0,0.0,0.0,0.0,0.0] --sim_type cosine --pooler_type avg --q_proj_type mlp --d_proj_type shared --temp 0.05 --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS  --warmup_steps 5000 --logging_steps 100 --eval_steps 5000 --save_steps 5000 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --dataloader_num_workers $NUM_WORKER --preprocessing_num_workers $NUM_WORKER --learning_rate 1e-5 --max_seq_length 512 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --fp16 --run_name $EXP_NAME > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid
