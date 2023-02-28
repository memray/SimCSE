export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.mtebeval.mteb_beir_eval.py --model-name-or-path intfloat/e5-small --pool-type avg --output-dir  /export/home/exp/search/mteb/e5-small --data-dir /export/home/data/beir
