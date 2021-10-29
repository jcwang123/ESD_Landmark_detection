import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

########################
from datasets.Landmarks import landmarks, landmarks_eval
from nets.resdcn import get_pose_net
########################

from utils.post_process import _topk,_nms
from utils.utils import _tranpose_and_gather_feature, load_model, load_model2
from utils.image import transform_preds
from utils.losses import _neg_loss, _reg_loss, _smooth_reg_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='/raid/wjc/logs/landmark_detection')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--folds', type=str)
parser.add_argument('--tag', type=str)
parser.add_argument('--planes', type=int)
parser.add_argument('--freeze_encoder', action='store_true')
parser.add_argument('--convert', action='store_true')


parser.add_argument('--arch', type=str)
parser.add_argument('--load_model', type=str)

parser.add_argument('--reg_loss', type=str, default='l1')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)
parser.add_argument('--scale', type=float, default=0.4)
parser.add_argument('--shift', type=float, default=0.4)



parser.add_argument('--t', type=int)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--gpus', type=str)

parser.add_argument('--use_relation', type=int)
parser.add_argument('--use_center', type=int)

parser.add_argument('--test_topk', type=int, default=100)


parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

os.makedirs(cfg.root_dir, exist_ok = True)


cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]

def main():
    for fold in cfg.folds.split(','):
        os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpus
        cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name, 'fold{}'.format(fold))
        cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name, 'fold{}'.format(fold))

        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
        logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
        summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
        print = logger.info
        print(cfg)
        torch.manual_seed(317)
        torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

        num_gpus = torch.cuda.device_count()
        if cfg.dist:
            cfg.device = torch.device('cuda:%d' % cfg.local_rank)
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://',
                                    world_size=num_gpus, rank=cfg.local_rank)
        else:
            cfg.device = torch.device('cuda')


        print('Setting up data...')
    
    
        train_dataset = landmarks(cfg.data_dir, 'train', fold=fold, img_size=cfg.img_size, scale=cfg.scale, shift=cfg.shift, use_relation=cfg.use_relation, use_center=cfg.use_center)
        val_dataset = landmarks_eval(cfg.data_dir, 'test', fold=fold)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                  num_replicas=num_gpus,
                                                  rank=cfg.local_rank)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                     batch_size=cfg.batch_size // num_gpus
                                     if cfg.dist else cfg.batch_size,
                                     shuffle=not cfg.dist,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     sampler=train_sampler if cfg.dist else None)
        print('Creating model...')
        if 'hourglass' in cfg.arch:
            model = get_hourglass[cfg.arch]

        elif 'resdcn_' in cfg.arch:
            from nets.resdcn import get_pose_net
            model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)

        elif 'dla' in cfg.arch:
            from nets.dla import get_pose_net as dla
            model = dla(num_layers=int(cfg.arch.split('_')[-1]), num_classes = train_dataset.num_classes)
            gcn = False
        else:
            raise NotImplementedError
        print('Model created!')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                   shuffle=False, num_workers=1, pin_memory=True,
                                   collate_fn=val_dataset.collate_fn)



        if cfg.load_model is not None:
            print('loading model:{}'.format(cfg.load_model))
            if cfg.load_model.split('/')[5] == 'CenterNet':
                model = load_model(model, cfg.load_model)
            elif cfg.load_model.split('/')[5] == 'simple_centernet':
                model = load_model2(model, cfg.load_model)
                
            else:
                model = load_model2(model, cfg.load_model)

        if cfg.dist:
            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(cfg.device)
            model = nn.parallel.DistributedDataParallel(model,
                                        device_ids=[cfg.local_rank, ],
                                        output_device=cfg.local_rank)
        else:
            model = nn.DataParallel(model).to(cfg.device)


        optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

        def train(epoch):
            print('\n Epoch: %d' % epoch)
            model.train()
            tic = time.perf_counter()
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=cfg.device, non_blocking=True)
                outputs = model(batch['image'])

                hmap, regs, w_h_ = zip(*outputs)
                hmap = torch.sigmoid(hmap[0])
                if cfg.convert:
                    hmap = convert_re(hmap)

                hmap_loss = _neg_loss([hmap], batch['hmap'])  
                regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]


                _reg_loss_f = {'sl1':_smooth_reg_loss,'l1':_reg_loss}
                reg_loss = _reg_loss_f[cfg.reg_loss](regs, batch['regs'], batch['ind_masks'])
                w_h_loss = _reg_loss_f[cfg.reg_loss](w_h_, batch['w_h_'], batch['ind_masks'])
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                          ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                          (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +
                          ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

                    step = len(train_loader) * epoch + batch_idx
                    summary_writer.add_scalar('Fold{}/hmap_loss'.format(fold), hmap_loss.item(), step)
                    summary_writer.add_scalar('Fold{}/reg_loss'.format(fold), reg_loss.item(), step)
                    summary_writer.add_scalar('Fold{}/w_h_loss'.format(fold), w_h_loss.item(), step)
            return

        def val_map(epoch):
            print('\n Val@Epoch: %d' % epoch)
            model.eval()
            torch.cuda.empty_cache()
            max_per_image = 100

            results = {}
            with torch.no_grad():
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

            eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
            print(eval_results)
            summary_writer.add_scalar('Fold{}/AP'.format(fold), eval_results[0], epoch)
            summary_writer.add_scalar('Fold{}/AP50'.format(fold), eval_results[1], epoch)
            summary_writer.add_scalar('Fold{}/AP75'.format(fold), eval_results[2], epoch)
            summary_writer.add_scalar('Fold{}/Recall'.format(fold), eval_results[-4], epoch)
            return eval_results[0]

        print('Starting training...')
        best = 0
        best_ep = 0
        for epoch in range(1, cfg.num_epochs + 1):
            train_sampler.set_epoch(epoch)
            train(epoch)
            if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
                save_map = val_map(epoch)
                if save_map>best:
                    best = save_map
                    best_ep = epoch
                    print(saver.save(model.module.state_dict(), 'checkpoint'))
                else:
                    if epoch-best_ep>30:
                        break
                print(saver.save(model.module.state_dict(), 'latestcheckpoint'))
            lr_scheduler.step(epoch)  # move to here after pytorch1.1.0
        summary_writer.close()

if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
