export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_RANK=0
export WORLD_SIZE=4

cd /export/share/ruimeng/project/search/simcse
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 --max_restarts=0 train.py --do_eval --arch_type moco --sim_type dot --queue_size 4096 --beir_datasets dbpedia-entity --model_name_or_path /export/home/exp/search/unsup_dr/exp_v1/pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5/checkpoints/checkpoint-200000/ --cache_dir /export/home/data/pretrain/.cache --per_device_eval_batch_size 64 --dataloader_num_workers 1 --preprocessing_num_workers 1 --fp16 --output_dir /export/home/exp/search/unsup_dr/exp_v1/pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5 --run_name pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5
