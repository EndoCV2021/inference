#!/bin/bash
source /home/dgmif/workspace/venv/endocv2021/bin/activate

python inference.py --cfg_path ./model/config.yaml \
                     --model_path ./model/model_0040999.pth \
                     --image_path ./sample_data \
                     --device cuda:0 

deactivate
