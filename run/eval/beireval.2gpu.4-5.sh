#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5
export LOCAL_RANK=0
export WORLD_SIZE=2

EXP_NAMES=(
#  "wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5"
#  "pile.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2"
#  "pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5"
#  "c4.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5"
#  "c4.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5"
#  "cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5"
  "wiki+cc.equal.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5"
)

MASTER_PORT=24455
#    QUEUE_SIZE=131072
QUEUE_SIZE=16384
#QUEUE_SIZE=4096
#QUEUE_SIZE=2048

echo $EXP_PATH

#datasets=(nq hotpotqa dbpedia-entity fever climate-fever quora msmarco)
datasets=(dbpedia-entity hotpotqa fever climate-fever msmarco)

cd /export/share/ruimeng/project/search/simcse

for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXP_PATH="/export/home/exp/search/unsup_dr/exp_v3/$EXP_NAME"
    CKPT_PATH="$EXP_PATH/checkpoints/checkpoint-200000/"
    for dataset in "${datasets[@]}"
    do
        echo "Evaluating $dataset"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE nohup python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT --max_restarts=0 train.py --do_eval --arch_type moco --queue_size $QUEUE_SIZE --sim_type dot --beir_datasets $dataset --model_name_or_path $CKPT_PATH --cache_dir /export/home/data/pretrain/.cache --per_device_eval_batch_size 128 --dataloader_num_workers 1 --preprocessing_num_workers 1 --fp16 --output_dir $EXP_PATH --run_name $EXP_NAME > $EXP_PATH/nohup_beireval.2gpu.4-5_$dataset.out 2>&1 &"
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE nohup python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT --max_restarts=0 train.py --do_eval --arch_type moco --queue_size $QUEUE_SIZE --sim_type dot --beir_datasets $dataset --model_name_or_path $CKPT_PATH --cache_dir /export/home/data/pretrain/.cache --per_device_eval_batch_size 128 --dataloader_num_workers 1 --preprocessing_num_workers 1 --fp16 --output_dir $EXP_PATH --run_name $EXP_NAME > $EXP_PATH/nohup_beireval.2gpu.4-5_$dataset.out 2>&1 &
        wait $!
        echo "Finished $dataset"
    done
done
echo "All done"
