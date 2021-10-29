import torch
import torch.nn as nn
from collections import OrderedDict


def _gather_feature(feat, ind, mask=None):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat


def _tranpose_and_gather_feature(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feature(feat, ind)
  return feat


def flip_tensor(x):
  return torch.flip(x, [3])
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
#   start_epoch = 0
#   state_dict_ = torch.load(model_path, map_location=lambda storage, loc: storage)
#   state_dict = OrderedDict()
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def load_model2(model, pretrain_dir):
  state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
  print('loaded pretrained weights form %s !' % pretrain_dir)
  state_dict = OrderedDict()

  # convert data_parallal to model
  for key in state_dict_:
    if key.startswith('module') and not key.startswith('module_list'):
      state_dict[key[7:]] = state_dict_[key]
    else:
      state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
  model_state_dict = model.state_dict()
  for key in state_dict:
    if key in model_state_dict:
#       print(key,state_dict[key].shape,model_state_dict[key].shape)
      if state_dict[key].shape != model_state_dict[key].shape:
        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
          key, model_state_dict[key].shape, state_dict[key].shape))
        state_dict[key] = model_state_dict[key]
    else:
      print('Drop parameter {}.'.format(key))
  for key in model_state_dict:
    if key not in state_dict:
      print('No param {}.'.format(key))
      state_dict[key] = model_state_dict[key]
  model.load_state_dict(state_dict, strict=False)

  return model


def count_parameters(model):
  num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
  print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=384):
  flops = []
  handles = []

  def conv_hook(self, input, output):
    flops.append(output.shape[2] ** 2 *
                 self.kernel_size[0] ** 2 *
                 self.in_channels *
                 self.out_channels /
                 self.groups / 1e6)

  def fc_hook(self, input, output):
    flops.append(self.in_features * self.out_features / 1e6)

  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      handles.append(m.register_forward_hook(conv_hook))
    if isinstance(m, nn.Linear):
      handles.append(m.register_forward_hook(fc_hook))

  with torch.no_grad():
    _ = model(torch.randn(1, 3, input_size, input_size))
  print("Total FLOPs = %f M" % sum(flops))

  for h in handles:
    h.remove()
