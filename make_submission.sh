#!/bin/bash
# source /Users/hyunseoki/workspace/venv/endocv2021/bin/activate

# python inference.py --cfg_path ./model/5.yaml \
#                      --model_path ./model/5.pth \
#                      --image_path ./sample_data \
#                      --device cpu 

# deactivate

mkdir EndoCV2021_detection
mkdir EndoCV2021_segmentation

mv ./EndoCV2021/detection ./EndoCV2021_detection
mv ./EndoCV2021/segmentation ./EndoCV2021_segmentation

zip -r EndoCV2021_detection.zip ./EndoCV2021_detection
zip -r EndoCV2021_segmentation.zip ./EndoCV2021_segmentation

rm -rf EndoCV2021 EndoCV2021_detection EndoCV2021_segmentation