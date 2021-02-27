#!/home/dgmif/workspace/venv/endocv2021/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021
@author: endocv2021@generalizationChallenge

Modified on Sat Feb 27 15:13:03 2021
@author: hyunseoki
"""

import os
import argparse
import numpy as np
import torch
import cv2
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
    
    task_type = 'segmentation'

    # set image folder here!
    directoryName = create_predFolder(task_type)
    
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3']
    
    for subDir in subDirs:
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        # imgfolder='/project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/' + subDirs[j]
        imgfolder = os.path.join(args.image_path, subDir)
        
        # set folder to save your checkpoints here!
        saveDir = os.path.join(directoryName , subDir +'_pred')
    
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')
      
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        file = open(saveDir + '/'+"timeElaspsed" + subDir +'.txt', mode='w')
        timeappend = []
    
        for imagePath in imgfiles:
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            print('filename is printing::=====>>', filename)

            img = cv2.imread(imagePath)

            start.record()
            outputs = predictor(img)
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))

            boxes = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            masks = outputs["instances"].to("cpu").pred_masks.numpy()

            if masks.shape[0] > 0:
                combined_mask = masks[0]

                for mask in masks[1:]:
                    combined_mask = np.bitwise_or(combined_mask, mask)    

            else:
                combined_mask = np.zeros_like(img)

            imsave(saveDir +'/'+ filename +'_mask.tif', (combined_mask*255.0).astype('uint8'))

            file.write('%s -----> %s \n' % 
               (filename, start.elapsed_time(end)))
        
        file.write('%s -----> %s \n' % 
           ('average_t', np.mean(timeappend)))