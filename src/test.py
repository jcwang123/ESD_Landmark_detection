import os
import time
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.utils.data
from torchsummary import summary

from datasets.Landmarks import landmarks, landmarks_eval


from nets.resdcn import get_pose_net
from nets.dla import get_pose_net as dla

from utils.utils import load_model, load_model2
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import ctdet_decode,_topk,_nms

from nms.nms import soft_nms

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='/raid/wjc/logs/landmark_detection')
parser.add_argument('--data_dir', type=str, default='/raid/wjc/code/ESD/Landmark_Detection/data')

parser.add_argument('--log_name', type=str)
parser.add_argument('--split', type=str,choices=['val','test'])
parser.add_argument('--convert', action='store_true')

parser.add_argument('--fold', type=str)
parser.add_argument('--arch', type=str, default='dla_34')
parser.add_argument('--load_model', type=str)

parser.add_argument('--use_relation', type=bool)
parser.add_argument('--use_center', type=bool)

parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_true')

parser.add_argument('--test_scales', type=str, default='0.5')    # 0.5,0.75,1,1.25,1.5

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name, 'fold{}'.format(cfg.fold))
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name, 'fold{}'.format(cfg.fold))
cfg.load_model = os.path.join(cfg.ckpt_dir, 'checkpoint.t7')
cfg.lstm = 'lstm' in cfg.arch
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]

def main():
    logger = create_logger(save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100

    from datasets.esd import ESD,ESD_eval
    val_dataset = landmarks_eval(cfg.data_dir, 'test', fold=cfg.fold, test_scales=cfg.test_scales, test_flip=False, fix_size=False)    

    if cfg.split is None:
        raise NotImplementedError
    print('Split:'+cfg.split)

    print('Creating model...')
    
    if 'dla' in cfg.arch:
        from nets.dla import get_pose_net as dla
        model = dla(num_layers=int(cfg.arch.split('_')[-1]), num_classes = 2 if cfg.use_relation else 1)
    else:
        raise NotImplementedError
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=1, pin_memory=True,collate_fn=val_dataset.collate_fn)        
        
#     val_loader = val_dataset.num_classes
    model = load_model2(model, cfg.load_model)
    model = model.to(cfg.device)
    
    model.eval()
    torch.cuda.empty_cache()
    max_per_image = 100

    results = {}
    with torch.no_grad():
        st = time.time()
        for inputs in val_loader:
            img_id, inputs = inputs[0]

            detections = []
            for scale in inputs:
                inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                hmap, regs, w_h_ = model(inputs[scale]['image'])[-1]
                hmap = torch.sigmoid(hmap)[:,:1]
                if cfg.convert:
                    hmap = convert_re(hmap)
                dets = ctdet_decode(hmap, regs, w_h_, K=cfg.test_topk, compute_edge_reg=False)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                top_preds = {}
                dets[:, :2] = transform_preds(dets[:, 0:2],
                                        inputs[scale]['center'],
                                        inputs[scale]['scale'],
                                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                         inputs[scale]['center'],
                                         inputs[scale]['scale'],
                                         (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                clses = dets[:, -1]
                for j in range(val_dataset.num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                           for j in range(1, val_dataset.num_classes + 1)}
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, val_dataset.num_classes + 1)])
            
            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, val_dataset.num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]
                
            results[img_id] = bbox_and_scores
        end = time.time()
        print((end-st)/643)

    eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
    print(eval_results)


if __name__ == '__main__':
    main()
