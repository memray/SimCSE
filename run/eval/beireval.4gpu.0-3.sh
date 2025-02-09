#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_RANK=0
export WORLD_SIZE=4

EXP_NAME="pile.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2"
EXP_PATH="/export/home/exp/search/unsup_dr/exp_v1/$EXP_NAME"
CKPT_PATH="$EXP_PATH/checkpoints/checkpoint-200000/"
MASTER_PORT=40123
QUEUE_SIZE=131072

echo $EXP_PATH

datasets=(quora nq dbpedia-entity hotpotqa fever climate-fever msmarco)
datasets=(fever climate-fever)

cd /export/share/ruimeng/project/search/simcse

for dataset in "${datasets[@]}"
do
    echo "Evaluating $dataset"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT --max_restarts=0 train.py --do_eval --arch_type moco --queue_size $QUEUE_SIZE --sim_type dot --beir_datasets $dataset --model_name_or_path $CKPT_PATH --cache_dir /export/home/data/pretrain/.cache --per_device_eval_batch_size 128 --dataloader_num_workers 1 --preprocessing_num_workers 1 --fp16 --output_dir $EXP_PATH --run_name $EXP_NAME > $EXP_PATH/nohup_beireval_$dataset.out 2>&1 &"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT --max_restarts=0 train.py --do_eval --arch_type moco --queue_size $QUEUE_SIZE --sim_type dot --beir_datasets $dataset --model_name_or_path $CKPT_PATH --cache_dir /export/home/data/pretrain/.cache --per_device_eval_batch_size 128 --dataloader_num_workers 1 --preprocessing_num_workers 1 --fp16 --output_dir $EXP_PATH --run_name $EXP_NAME > $EXP_PATH/nohup_beireval.4gpu.0-3_$dataset.out 2>&1 &
    wait $!
    echo "Finished $dataset"
done
echo "All done"
