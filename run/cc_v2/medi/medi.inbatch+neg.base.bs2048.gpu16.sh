#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LOCAL_RANK=0
export WORLD_SIZE=16

export TOKENIZERS_PARALLELISM=false
export NUM_WORKER=5
export MAX_STEPS=100000

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

export EXP_NAME=medi.inbatch+neg.inbatch.bert-base-uncased.avg.dot.q128d128.step100k.bs2048.lr5e5
export PROJECT_DIR=/export/home/exp/search/unsup_dr/cc_v2/$EXP_NAME
mkdir -p $PROJECT_DIR
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export HF_DATASETS_CACHE=/export/home/data/pretrain/.cache
export TRANSFORMERS_CACHE=/export/home/cache/hf
export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=e276a36a5641a49d67b9bf4e9b48b849deffaa76
export WANDB_PROJECT=unsup_retrieval_cc
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb

cd /export/home/project/search/uir_best_cc
nohup python -m torch.distributed.launch --nproc_per_node=16 --master_port=31133 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type inbatch --pooling avg --q_proj none --k_proj none --data_type medi --train_file /export/home/data/pretrain/medi/medi-data.jsonl --neg_names neg_docs --hard_negative_num 1 --dev_file /export/home/data/pretrain/wiki2021_structure/wiki_psgs_w100.dev.tail2e13.tsv --remove_unused_columns False --sim_type dot --projection_size 768 --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 10000 --logging_steps 100 --eval_steps 10000 --save_steps 1000000 --per_device_train_batch_size 128 --per_device_eval_batch_size 128  --beir_batch_size 128 --qa_batch_size 32 --dataloader_num_workers $NUM_WORKER --lr_scheduler_type linear --learning_rate5e-5  --max_seq_length 256 --max_q_tokens 128 --max_d_tokens 128 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 42 --report_to wandb --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid

#nohup python -m torch.distributed.launch --nproc_per_node=16 --master_port=31133 --max_restarts=0 train.py --model_name_or_path bert-large-uncased --arch_type moco --train_file /export/home/data/search/upr/cc/T03B_PileCC_topic.json --dev_file /export/home/data/pretrain/wiki2021_structure/wiki_psgs_w100.dev.tail2e13.tsv --data_type hf --data_pipeline_name contriever256-special50% --remove_unused_columns False --sim_type dot --queue_size 16384 --projection_size 1024 --momentum 0.9995 --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 10000 --logging_steps 100 --eval_steps 10000 --save_steps 1000000 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --beir_batch_size 256 --dataloader_num_workers $NUM_WORKER --learning_rate 5e-5 --max_seq_length 192 --max_q_tokens 4 64 --max_d_tokens 128 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 42 --report_to wandb --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid

#export WANDB_RUN_ID=13h4vrrq
#nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=3112 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type inbatch --reload_model_from $PROJECT_DIR --remove_unused_columns False --sim_type dot --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --do_eval --skip_senteval --skip_qaeval --beir_datasets robust04 bioasq signal1m trec-news --run_name $EXP_NAME --fp16 --seed 477 --report_to wandb --wiki_passage_path /export/home/data/search/nq/psgs_w100.tsv --qa_datasets_path /export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup-eval.log 2>&1 & echo $! > run.pid

