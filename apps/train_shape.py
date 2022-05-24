import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index
#pcs_fuda require
from pcs.run import pcs_run
#tSNE
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

#k means
from sklearn.cluster import KMeans

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import math

# get options
opt = BaseOptions().parse()

def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)      

    return torch.mean(XX + YY - 2. * XY)

def train(opt):
    
    writer = SummaryWriter('./log/%s' % opt.name)
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_source_dataset = TrainDataset(opt, phase='train')
    train_target_dataset = TrainDataset(opt, phase='train', is_sculpture=True)

    projection_mode = 'orthogonal'
    
    train_batch_size = opt.batch_size

    train_source_loader = DataLoader(train_source_dataset,
                                     batch_size=train_batch_size,
                                     shuffle=not opt.serial_batches,
                                     num_workers=opt.num_threads,
                                     pin_memory=opt.pin_memory)

    print('train source loader size: ', len(train_source_loader))#720    
    
    train_target_loader = DataLoader(train_target_dataset,
                                     batch_size=train_batch_size,
                                     shuffle=not opt.serial_batches,
                                     num_workers=opt.num_threads,
                                     pin_memory=opt.pin_memory
                                     )
    print('train target data size: ', len(train_target_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)    
    print('Using Network: ', netG.name)

    if opt.load_netG_checkpoint_path is not None:
        print('loading for netG...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            modelG_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            modelG_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming netG from ', modelG_path)
        netG.load_state_dict(torch.load(modelG_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))
    
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    
    train_source_iter = iter(train_source_loader)
    train_target_iter = iter(train_target_loader)
    
    mse = nn.MSELoss()
    
    for epoch in range(start_epoch, opt.num_epoch):
        
        epoch_start_time = time.time()
        num_batches = 10
        
        netG.train()

        iter_data_time = time.time()
        for batch_i in range(num_batches):
            
            iter_start_time = time.time()
            
            # retrieve the source train data
            train_source_data = next(train_source_iter, -1)
            if isinstance(train_source_data, int):
                train_source_iter = iter(train_source_loader)
                train_source_data = next(train_source_iter, -1)
            train_source_image_tensor = train_source_data['img'].to(device=cuda)
            train_source_calib_tensor = train_source_data['calib'].to(device=cuda)
            train_source_image_tensor, train_source_calib_tensor = reshape_multiview_tensors(train_source_image_tensor, train_source_calib_tensor)
            train_source_sample_tensor = train_source_data['samples'].to(device=cuda)
            train_source_label_tensor = train_source_data['labels']
            train_source_label_tensor = train_source_label_tensor.permute(0,2,1).reshape(-1,1).squeeze()

            # retrieve the target train data
            train_target_data = next(train_target_iter, -1)
            if isinstance(train_target_data, int):
                train_target_iter = iter(train_target_loader)
                train_target_data = next(train_target_iter, -1)
            train_target_image_tensor = train_target_data['img'].to(device=cuda)
            train_target_calib_tensor = train_target_data['calib'].to(device=cuda)
            train_target_image_tensor, train_target_calib_tensor = reshape_multiview_tensors(train_target_image_tensor, train_target_calib_tensor)
            train_target_sample_tensor = train_target_data['samples'].to(device=cuda)
            train_target_label_tensor = train_target_data['labels']
            train_target_label_tensor = train_target_label_tensor.permute(0,2,1).reshape(-1,1).squeeze()
            loss = 0
            if True:
                train_image_tensor = torch.cat((train_source_image_tensor,train_target_image_tensor), dim = 0)
                train_calib_tensor = torch.cat((train_source_calib_tensor,train_target_calib_tensor), dim = 0)
                train_sample_tensor = torch.cat((train_source_sample_tensor,train_target_sample_tensor), dim = 0)
                # train_label_tensor = torch.cat((train_source_label_tensor, train_target_label_tensor), dim = 0)
                
                _, train_feature_list, train_preds_list = netG.forward(train_image_tensor, train_sample_tensor, train_calib_tensor)
                
                for n_stack in range(opt.num_stack):
                    
                    train_feature = train_feature_list[n_stack]
                    train_feature = train_feature.permute(0,2,1).reshape(-1,train_feature.shape[1])
                    train_source_feature = train_feature[:train_batch_size*opt.num_sample_inout,:]
                    train_target_feature = train_feature[train_batch_size*opt.num_sample_inout:,:]
                    
                    loss_mmd = MMD(train_source_feature[:,:], train_target_feature[:,:], "rbf", device=cuda)
                    
                    train_preds = train_preds_list[n_stack].permute(0,2,1).reshape(-1,1).squeeze()
                    train_source_result = train_preds[:train_batch_size*opt.num_sample_inout]
                    train_target_result = train_preds[train_batch_size*opt.num_sample_inout:]
                    
                    loss_source_mse = mse(train_source_result, train_source_label_tensor)

                    w_mmd = 5
                    w_s_mse = 2
                    w_t_mse = max(0,(epoch - start_epoch) / opt.num_epoch)
                    w_mi = -max(0,(epoch - start_epoch) / opt.num_epoch)                   

                    loss = loss + w_mmd * loss_mmd + w_s_mse * loss_source_mse

		train_target_feature[:,-1] = train_target_feature[:,-1]*256
		train_source_feature[:,-1] = train_source_feature[:,-1]*256
		train_cdist = torch.cdist(train_target_feature, train_source_feature, p=2)

		_, idx = torch.topk(train_cdist, k = int(opt.num_sample_inout/100), dim = 1, largest = False)
                        
		prediction = train_source_label_tensor.expand(idx.shape[0], -1)
		ind = np.indices(idx.shape)
                   ind[-1] = idx
                   prediction = prediction[tuple(ind)]#selected predicted results
                        
                   prediction = prediction.sum(dim = 1) / idx.shape[1]
                        
                   m = max(0,(epoch - start_epoch) / opt.num_epoch)
		train_target_pseudo_label = m * train_target_result.cpu() + (1-m) * prediction.cpu()
                        
                   loss_target_mse = nn.MSELoss()(train_target_result, train_target_pseudo_label)
                   loss = loss + w_t_mse * loss_target_mse
                        
                    softmax_target_result = torch.softmax(torch.cat((1-train_target_result.unsqueeze(1), train_target_result.unsqueeze(1)), dim = -1), dim = 1)
                    mean_target_result = torch.mean(softmax_target_result, dim = 0)
                    loss_mi_1 = -(mean_target_result * torch.log(mean_target_result)).sum()
                    loss_mi_2 = -(softmax_target_result * torch.log(softmax_target_result)).sum() / softmax_target_result.shape[0]
                    loss_target_mi = loss_mi_1 - loss_mi_2
                    loss = loss + w_mi * loss_target_mi
                        
                
                loss = loss / opt.num_stack
                print(f'loss: {loss}')
                optimizerG.zero_grad()
                loss.backward()
                optimizerG.step()
                
            writer.add_scalar('loss', loss.item(), epoch)
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (batch_i + 1)) * num_batches - (
                    iter_net_time - epoch_start_time)

            if batch_i % opt.freq_plot == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Loss: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netG: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        opt.name, epoch, batch_i, num_batches, loss.item(), lr, opt.sigma,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60))))
                
            iter_data_time = time.time()

        torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
        torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))


if __name__ == '__main__':
    train(opt)
