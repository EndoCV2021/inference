#!/home/dgmif/workspace/venv/endocv2021/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021
@author: endocv2021@generalizationChallenge

Modified on Sun Feb 28 17:18:03 2021
@author: hyunseoki

Modified on Thu March 2 14:18:03 2021
@author: Doyoob Yeo (ensemble inference)
"""

import os
import argparse
from glob import glob

import numpy as np
import copy
import cv2
import json

import torch
from tifffile import imsave
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from ensemble_boxes import weighted_boxes_fusion


def create_predFolder(task_type):
    directoryName = 'EndoCV2021'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
        
    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))
        
    return os.path.join(directoryName, task_type)


def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, default='./model',
                        help="PATH TO DETECTRON2 CONFIG")
    parser.add_argument("--model_dir", type=str, default='./model',
                        help="PATH TO MODEL WEIGHT")
    parser.add_argument("--image_path", type=str, default='./project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/',
                        help="PATH TO IMAGE FOLDER")
    parser.add_argument("--device", type=str, default='cuda',
                        help="DEVICE ID")

    return parser


def get_predictor(cfg_path, model_path, device):
    '''
    Returns
    -------
    model : detectron2.engine.defaults.DefaultPredictor
        DESCRIPTION.
    device : str
        DESCRIPTION.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device    

    return DefaultPredictor(cfg)

coco_format = {
    "images": [
    ],
    "categories": [
        {"supercategory": "none", 
         "id": 1, 
         "name": "polyp"}
         ], 
    "annotations": [
    ]
}

def create_image_annotation(file_name, height, width, image_id):
    file_name = file_name.split('/')[-1] 
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


def create_annotation_coco_format(x1, y1, x2, y2, score, image_id, annotation_id):
    min_x = float(x1)
    min_y = float(y1)
    width = float(x2 - x1)
    height = float(y2 - y1)
    area = width * height
    bbox = (min_x, min_y, width, height)   

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': 1,
        'segmentation': [],
        'score': float(score)
    }

    return annotation


if __name__ == '__main__':
    args = get_argparser().parse_args()     

    cfg_list = [os.path.basename(x) for x in glob(os.path.join(args.cfg_dir, '*.yaml'))]
    cfg_list = sorted(cfg_list)[:5]

    pth_list = [os.path.basename(x) for x in glob(os.path.join(args.model_dir, '*.pth'))]
    pth_list = sorted(pth_list)

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    # set image folder here!
    segDirectoryName = create_predFolder('segmentation')
    detDirectoryName = create_predFolder('detection')
    
    subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3']
    # settings for weighted_boxes_fusion
    # referred at https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    iou_thr = 0.5
    skip_box_thr = 0.1
    sigma = 0.1
    weights = [1] * 5

    for subDir in subDirs:
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        # imgfolder='/project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/' + subDirs[j]
        imgfolder = os.path.join(args.image_path, subDir)
        
        # set folder to save your checkpoints here!
        segSaveDir = os.path.join(segDirectoryName , subDir +'_pred')        
    
        if not os.path.exists(segSaveDir):
            os.mkdir(segSaveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')

        if args.device != 'cpu':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            file = open(segSaveDir + '/'+"timeElaspsed" + subDir +'.txt', mode='w')
            timeappend = []

        cocoInstance = copy.deepcopy(coco_format)
        annotations_id = 2 ## start from 2

        for image_id, imagePath in enumerate(imgfiles):
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            print('filename is printing::=====>>', filename)

            img = cv2.imread(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # initialize variables for saving results
            boxes_list = []
            scores_list = []
            labels_list = []
            seg_masks = np.zeros((img.shape[0], img.shape[1]))

            for k in range(5):
                cfg_path = os.path.join(args.cfg_dir, cfg_list[k])
                pth_path = os.path.join(args.model_dir, pth_list[k])
                predictor = get_predictor(
                    cfg_path=cfg_path,
                    model_path=pth_path,
                    device=args.device
                )

                if args.device != 'cpu':
                    start.record()
                    outputs = predictor(img)
                    end.record()
                    torch.cuda.synchronize()
                    # print(start.elapsed_time(end))
                    timeappend.append(start.elapsed_time(end))
                else:
                    outputs = predictor(img)

                instances = outputs['instances'].to('cpu')
                image_height, image_width = instances.image_size
                pred_boxes = instances.pred_boxes.tensor.numpy().tolist()
                pred_scores = instances.scores.numpy().tolist()
                pred_classes = instances.pred_classes.numpy().tolist()

                pred_masks = instances.pred_masks.numpy()
                pred_masks = np.sum(pred_masks, axis=0)

                # normalize coordinates of bounding boxes
                # weighted_boxes_fusion library를 사용하기 위해서는 bbox 좌표값을 0~1 사이값으로 normalization 해줘야 한다
                for pred_box in pred_boxes:
                    pred_box[0] = pred_box[0] / image_width
                    pred_box[1] = pred_box[1] / image_height
                    pred_box[2] = pred_box[2] / image_width
                    pred_box[3] = pred_box[3] / image_height

                # detect된 객체가 있는 경우에만 저장
                if len(outputs) > 0:
                    boxes_list.append(pred_boxes)
                    scores_list.append(pred_scores)
                    labels_list.append(pred_classes)

                seg_masks += pred_masks

            # 모델의 결과들을 모은 결과로부터 ensemble 하기

            # segmentation 결과 ensemble
            seg_masks = seg_masks * 1.0 / 5.0
            seg_masks[seg_masks < 0.5] = 0
            seg_masks[seg_masks > 0.5] = 1
            combined_mask = seg_masks.astype(np.double)

            # bounding box 결과 ensemble
            if boxes_list is not None:
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
                )
            else:
                boxes = []
                scores = []
                labels = []

            # detectron 결과는 절대 좌표값으로 저장해야 되므로 w, h값을 곱해준다
            for box in boxes:
                box[0] = box[0] * image_width
                box[1] = box[1] * image_height
                box[2] = box[2] * image_width
                box[3] = box[3] * image_height


            ####### SEGMENTATION #######
            # if masks.shape[0] > 0:
            #     combined_mask = masks[0]
            #
            #     for mask in masks[1:]:
            #         combined_mask = np.bitwise_or(combined_mask, mask)
            #
            # else:
            #     combined_mask = np.zeros_like(img)

            imsave(segSaveDir +'/'+ filename +'_mask.jpg', (combined_mask*255.0).astype('uint8'))
                 
            ######## DETECTION #########
            cocoInstance['images'].append(
                create_image_annotation(
                    file_name=imagePath.split('/')[-1],
                    height=image_height, 
                    width=image_width,
                    image_id=image_id
                    )
                )
            for box, score in zip(boxes, scores):
                cocoInstance['annotations'].append(
                    create_annotation_coco_format(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        score=score,
                        image_id=image_id,
                        annotation_id=annotations_id)
                    )
    
                annotations_id += 1

            if args.device != 'cpu':
                file.write('%s -----> %s \n' % 
                    (filename, start.elapsed_time(end)))

        segResultFN = os.path.join(detDirectoryName, subDir + '.json')
        
        with open(segResultFN, 'w') as f:
            json.dump(cocoInstance, f)

        if args.device != 'cpu':
            file.write('%s -----> %s \n' % 
                ('average_t', np.mean(timeappend)))