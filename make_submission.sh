source /Users/hyunseoki/workspace/venv/endocv2021/bin/activate                                              Modified
# source ../env01/bin/activate
pip install -r requirements.txt > /dev/null

echo ======= Detection Start =======

python -W ignore inference_ensemble_detection.py \
                     --cfg_dir ensemble_weights/segmentation \
                     --model_dir ensemble_weights/segmentation \
                     --image_path ./sample_data \
                     --device cpu 

zip -r result_detection.zip EndoCV2021
rm -rf EndoCV2021

echo ======= Segmentation Start =======
python -W ignore inference_ensemble_segmentation.py \
                     --cfg_dir ensemble_weights/detection \
                     --model_dir ensemble_weights/detection \
                     --image_path ./sample_data \
                     --device cpu 

zip -r result_segmentation.zip EndoCV2021
rm -rf EndoCV2021

deactivate
