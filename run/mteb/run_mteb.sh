export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.mtebeval.mteb_augtriever.py
