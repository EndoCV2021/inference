#!/home/dgmif/workspace/venv/endocv2021/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021
@author: endocv2021@generalizationChallenge

Modified on Sun Feb 28 17:18:03 2021
@author: hyunseoki
"""

import os
import argparse
import numpy as np
import copy
import cv2
import json
import torch
from tifffile import imsave
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


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
    parser.add_argument("--cfg_path", type=str, default='./model/Mask_RCNN_Res50_FPN_3x.yaml',
                        help="PATH TO DETECTRON2 CONFIG")
    parser.add_argument("--model_path", type=str, default='./model/model_0040999.pth',
                        help="PATH TO MODEL WEIGHT")
    parser.add_argument("--image_path", type=str, default='./sample_data',
                        help="PATH TO IMAGE FOLDER")
    parser.add_argument("--device", type=str, default='cpu', 
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

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    predictor = get_predictor(
        cfg_path=args.cfg_path,
        model_path=args.model_path,
        device=args.device
        )
    
    # set image folder here!
    segDirectoryName = create_predFolder('segmentation')
    detDirectoryName = create_predFolder('detection')
    
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3']
    
    for subDir in subDirs:
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        # imgfolder='/project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/' + subDirs[j]
        imgfolder = os.path.join(args.image_path, subDir)
        
        # set folder to save your checkpoints here!
        segSaveDir = os.path.join(segDirectoryName , subDir +'_pred')        
    
        if not os.path.exists(segSaveDir):
            os.mkdir(segSaveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')
      
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

            start.record()
            outputs = predictor(img)
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))

            instances = outputs['instances'].to('cpu')
            image_height, image_width = instances.image_size
            boxes = instances.pred_boxes.tensor.numpy() ## return float matrix of Nx4. Each row is (x1, y1, x2, y2).
            scores = instances.scores.numpy()
            masks = instances.pred_masks.numpy()

            ####### SEGMENTATION #######
            if masks.shape[0] > 0:
                combined_mask = masks[0]

                for mask in masks[1:]:
                    combined_mask = np.bitwise_or(combined_mask, mask)    

            else:
                combined_mask = np.zeros_like(img)

            imsave(segSaveDir +'/'+ filename +'_mask.tif', (combined_mask*255.0).astype('uint8'))
                 
            ######## DETECTION #########
            cocoInstance['images'].append(
                create_image_annotation(
                    file_name=filename,
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

            file.write('%s -----> %s \n' % 
               (filename, start.elapsed_time(end)))

        segResultFN = os.path.join(detDirectoryName, subDir + '.json')
        
        with open(segResultFN, 'w') as f:
            json.dump(cocoInstance, f)
       
        file.write('%s -----> %s \n' % 
           ('average_t', np.mean(timeappend)))