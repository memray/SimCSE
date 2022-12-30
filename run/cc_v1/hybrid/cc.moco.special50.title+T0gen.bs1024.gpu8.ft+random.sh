#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_RANK=0
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=false
export NUM_WORKER=4
export MAX_STEPS=20000

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export EXP_NAME=FT-inbatch-random-neg1023+1024.cc-hybrid.RC50+title10+T0gen40.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5
export ORI_EXP_PATH=/export/home/exp/search/unsup_dr/cc_v1/cc-hybrid.RC50+title10+T0gen40.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/wikipsg_v1-FT/$EXP_NAME
mkdir -p $PROJECT_DIR
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=c338136c195ab221b8c7cfaa446db16b2e86c6db
export WANDB_PROJECT=unsup_retrieval_wikipsg-FT
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb

cd /export/home/project/search/uir_best_cc
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=31133 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --reload_model_from $ORI_EXP_PATH --arch_type inbatch --data_pipeline_name finetune --data_type finetune --neg_names neg_docs --negative_strategy random --train_file /export/home/data/search/msmarco/msmarco-bm25.jsonl --dev_file /export/home/data/pretrain/wiki2021_structure/wiki_psgs_w100.dev.tail2e13.tsv --remove_unused_columns False --sim_type dot --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 1000 --logging_steps 100 --eval_steps 5000 --save_steps 100000 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --beir_batch_size 256 --dataloader_num_workers $NUM_WORKER --learning_rate 1e-5 --max_q_tokens 192 --max_d_tokens 192 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 577 --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json --report_to wandb > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid
