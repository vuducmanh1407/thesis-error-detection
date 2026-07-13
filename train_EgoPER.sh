#!/bin/bash
source activate main_env

pip install jsonlines

CUDA_VISIBLE_DEVICES=0 python main.py --config configs/EgoPER/tea/vc_4omini_post_db0.6.json --dir best
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/EgoPER/oatmeal/vc_4omini_post_db0.4.json --dir best
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/EgoPER/pinwheels/vc_4omini_post_db0.3.json --dir best
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/EgoPER/quesadilla/vc_4omini_post_db0.2.json --dir best
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/EgoPER/coffee/vc_4omini_post_db0.3_ndb0.json --dir best --vis