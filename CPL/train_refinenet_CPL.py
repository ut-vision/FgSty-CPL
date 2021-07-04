import argparse
import os, sys
import itertools
import os.path as osp
import pprint
import random
import copy
import warnings
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from refinenet import rf101
from discriminator import netD_pixel, grad_reverse
from hand import HandDataset
from utils import AverageMeter, prob2seg, save_log, pred2score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

# assert consecutive gpus are available (default: gpu0 and gpu1)
# Training settings
parser = argparse.ArgumentParser(description='refinenet')
parser.add_argument('--dataset', help='facades', default="/path/to/your/style-adapted-dataset")
parser.add_argument('--src_dataset', help='facades', default="/path/to/your/source-dataset")
parser.add_argument('--trg_dataset', help='facades', default="/path/to/your/target-dataset")
parser.add_argument('--batchSize', type=int, default=5, help='training batch size')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs')
parser.add_argument('--epoch_start', type=int, default=1, help='when to start epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--eta', type=float, default=0.8, help='eta for GRL')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--use_all', action='store_true', help='use all dataset?')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--iou_thresh', type=float, default=0.8, help='threshold for PL')
parser.add_argument('--model_path', type=str, help='whether to load a on-going model')
parser.add_argument('--src_model_path', type=str, help='path to source model')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
opt = parser.parse_args()
if opt.gpu >= 0:
    opt.cuda = True
opt.dataset = opt.dataset.rstrip("/")
name = opt.dataset.split("/")[-1] + "_CPL_alpha_%s_beta_%s_" % (str(opt.iou_thresh), str(opt.eta))
name += str(opt.seed)

print(name)
os.makedirs("results/%s"%(name), exist_ok=True)
save_log(str(datetime.now()), "results/%s/out.txt" % (name))
save_log(str(opt), "results/%s/out.txt" % (name))

np.random.seed(0)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    gpu_ids = [opt.gpu]
    cudnn.deterministic = True
    cudnn.benchmark = False
else:
    gpu_ids = []

sty_source_dataset = HandDataset(opt.dataset, "all")
source_dataset = HandDataset(opt.src_dataset, "all")
target_dataset = HandDataset(opt.trg_dataset, "train")

sty_source_loader = DataLoader(dataset=sty_source_dataset, batch_size=opt.batchSize, shuffle=True)
source_loader = DataLoader(dataset=source_dataset, batch_size=opt.batchSize, shuffle=True)
target_loader = DataLoader(dataset=target_dataset, batch_size=opt.batchSize, shuffle=True)

model_src = rf101(opt.output_nc, pretrained=False, use_dropout=True)
model_src.load_state_dict(torch.load(opt.src_model_path, map_location="cuda:%d"%opt.gpu))
if opt.model_path:
    model = rf101(opt.output_nc, pretrained=False, use_dropout=True)
    model.load_state_dict(torch.load(opt.model_path))
    print("loaded model from", opt.model_path)
else:
    model = rf101(opt.output_nc, pretrained=True, model_path="pretrained_models/refinenet101_voc.pth.tar", use_dropout=True)
    print("train from a voc pretrained model")

if opt.eta > 0:
    netD = netD_pixel()
else:
    netD = None

criterionBCE1 = nn.BCELoss()
criterionBCE2 = nn.BCELoss()
if opt.cuda:
    model = model.cuda(device=opt.gpu)
    model_src = model_src.cuda(device=opt.gpu + 1)
    if opt.eta > 0:
        netD = netD.cuda(device=opt.gpu)
    criterionBCE1 = criterionBCE1.cuda(device=opt.gpu)
    criterionBCE2 = criterionBCE2.cuda(device=opt.gpu+1)

iter_limit = 5000
# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
if opt.eta > 0:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr/5, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    optimizerD2 = optim.Adam(netD.cuda(device=opt.gpu + 1).parameters(), lr=opt.lr/5, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizer_src = optim.Adam(model_src.parameters(), lr=opt.lr/2, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

def train_one_epoch(model, model_src, netD, epoch, sty_source_loader, source_loader, target_loader):
    model.train()
    if opt.eta > 0:
        model_src.train()
    train_losses = AverageMeter()
    train_PL_losses = AverageMeter()
    train_adv_s_losses = AverageMeter()
    train_adv_t_losses = AverageMeter()
    train_iou = AverageMeter()
    train_f1 = AverageMeter()
    n_iters = 0
    data_iter_s = iter(source_loader)
    data_iter_ss = iter(sty_source_loader)
    data_iter_t = iter(target_loader)
    PL_pool = {"img": [], "PL": [], "gt": []}
    for i in range(iter_limit//opt.batchSize):
        try:
            sample_s = next(data_iter_s)
        except StopIteration:
            data_iter_s = iter(source_loader)
            sample_s = next(data_iter_s)
        try:
            sample_ss = next(data_iter_ss)
        except StopIteration:
            data_iter_ss = iter(sty_source_loader)
            sample_ss = next(data_iter_ss)
        try:
            sample_t = next(data_iter_t)
        except StopIteration:
            data_iter_t = iter(target_loader)
            sample_t = next(data_iter_t)
        img_s = sample_s['img']
        img_ss = sample_ss['img']
        img_t = sample_t['img']
        mask_s = sample_s['mask'].float()
        mask_ss = sample_ss['mask'].float()
        mask_t = sample_t['mask'].float()
        n_iters += opt.batchSize
        # forward
        pred_ss, feat_ss = model(img_ss.cuda(device=opt.gpu), with_feat=True)
        pred_t1, feat_t1 = model(img_t.cuda(device=opt.gpu), with_feat=True)
        if opt.eta > 0:
            pred_s, feat_s = model_src(img_s.cuda(device=opt.gpu+1), with_feat=True)
            pred_t2, feat_t2 = model_src(img_t.cuda(device=opt.gpu+1), with_feat=True)
        else:
            with torch.no_grad():
                pred_t2 = model_src(img_t.cuda(device=opt.gpu+1), with_feat=False)

        # PL
        scores_list = pred2score(pred_t2, pred_t1, out_list=True)
        assert len(scores_list["miou"]) == img_t.size(0)
        PL_loss1 = torch.zeros(1)
        for j, (iou, intersection) in enumerate(zip(scores_list["miou"], scores_list["intersection"])):
            if iou > opt.iou_thresh:
                PL_pool["gt"].append(mask_t[j])
                PL_pool["img"].append(img_t[j])
                PL_pool["PL"].append(torch.from_numpy(intersection.astype(np.float32).transpose(2, 0, 1)))
        if len(PL_pool["gt"]) > 0:
            if len(PL_pool["gt"]) >= opt.batchSize:
                pred_PL = model(torch.stack(PL_pool["img"][:opt.batchSize]).cuda(device=opt.gpu))
                PL_loss1 = criterionBCE1(pred_PL, torch.stack(PL_pool["PL"][: opt.batchSize]).cuda(device=opt.gpu))

        optimizer.zero_grad()
        if opt.eta > 0:
            optimizerD.zero_grad()
            netD = netD.cuda(device=opt.gpu)   
        # seg loss
        loss = criterionBCE1(pred_ss, mask_ss.cuda(device=opt.gpu))
        # adv
        if opt.eta > 0:
            disc_feat_ss = netD(grad_reverse(feat_ss, opt.eta))
            disc_feat_t1 = netD(grad_reverse(feat_t1, opt.eta))
            disc_loss_ss = torch.mean(disc_feat_ss ** 2) * 0.5 #0: SA
            disc_loss_t1 = torch.mean((1 - disc_feat_t1) ** 2) * 0.5  #1: MT
            loss += (disc_loss_ss + disc_loss_t1)
        # cpl
        if PL_loss1:
            loss += PL_loss1
            train_PL_losses.update(PL_loss1.data.item(), len(PL_pool["gt"]))
            del PL_pool
            PL_pool = {"img": [], "PL": [], "gt": []}
        loss.backward()
        optimizer.step()
        if opt.eta > 0:
            optimizerD.step()
        
        if opt.eta > 0:
            optimizer_src.zero_grad() 
            optimizerD2.zero_grad()
            loss2 = criterionBCE2(pred_s, mask_s.cuda(device=opt.gpu + 1))
            # adv
            netD = netD.cuda(device=opt.gpu + 1)
            disc_feat_s = netD(grad_reverse(feat_s, opt.eta))
            disc_feat_t2 = netD(grad_reverse(feat_t2, opt.eta))
            disc_loss_s = torch.mean(disc_feat_s ** 2) * 0.5 #0: S
            disc_loss_t2 = torch.mean((1 - disc_feat_t2) ** 2) * 0.5  #1: MT
            loss2 += (disc_loss_s + disc_loss_t2)
            loss2.backward()
            optimizer_src.step()
            optimizerD2.step()
            netD = netD.cuda(device=opt.gpu)

        train_losses.update(loss.data.item(), opt.batchSize)
        if opt.eta > 0:
            train_adv_s_losses.update(disc_loss_ss.data.item(), opt.batchSize)
            train_adv_t_losses.update(disc_loss_t1.data.item(), opt.batchSize)

        # score
        _iou, _f1 = pred2score(pred_ss, mask_ss)
        if _iou:
            train_iou.update(_iou, opt.batchSize)
            train_f1.update(_f1, opt.batchSize)

        if (i+1) % 100 ==0 or i==len(sty_source_loader)-1:
            save_log('Train: [{0}][{1}/{2}]\tLoss {loss_g.avg:.4f}\tAdv_s {loss_adv_s.avg:.4f}\tAdv_t {loss_adv_t.avg:.4f}\tPL {loss_PL.avg:.4f}'.format(
                epoch, i + 1, len(sty_source_loader), loss_g=train_losses, loss_adv_s=train_adv_s_losses, loss_adv_t=train_adv_t_losses, loss_PL=train_PL_losses), "results/%s/out.txt" % (name))

for epoch in range(opt.epoch_start, opt.nEpochs):
    train_one_epoch(model, model_src, netD, epoch, sty_source_loader, source_loader, target_loader)
    torch.save(model.state_dict(), "results/%s/rf101_%03d.pth" % (name, epoch))
