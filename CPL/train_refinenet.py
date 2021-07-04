import argparse
import os, sys
import os.path as osp
import pprint
import random
import warnings
from datetime import datetime

import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from refinenet import rf101
from hand import HandDataset
from utils import AverageMeter, prob2seg, save_log, pred2score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='refinenet')
parser.add_argument('--dataset', help='facades', default="/path/to/your/dataset")
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs')
parser.add_argument('--epoch_start', type=int, default=1, help='when to start epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--use_all', action='store_true', help='use all dataset?')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, help='whether to load a on-going model')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
opt = parser.parse_args()
if opt.gpu >= 0:
    opt.cuda = True

opt.dataset = opt.dataset.rstrip("/")
name = opt.dataset.split("/")[-1] + "_" + str(opt.seed)
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

trainset = HandDataset(opt.dataset, "train")
testset = HandDataset(opt.dataset, "test")
train_loader = DataLoader(dataset=trainset, batch_size=opt.batchSize, shuffle=True)

if opt.model_path:
    model = rf101(opt.output_nc, pretrained=False, use_dropout=True)
    model.load_state_dict(torch.load(opt.model_path))
    print("loaded model from", opt.model_path)
else:
    model = rf101(opt.output_nc, pretrained=True, model_path="pretrained_models/refinenet101_voc.pth.tar", use_dropout=True)
    print("train from a voc pretrained model")
criterionBCE = nn.BCELoss()
if opt.cuda:
    model = model.cuda(device=opt.gpu)
    criterionBCE = criterionBCE.cuda(device=opt.gpu)


# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

def train_one_epoch(model, train_loader):
    model.train()
    train_losses = AverageMeter()
    train_iou = AverageMeter()
    train_f1 = AverageMeter()
    n_iters = 0
    for i, sample in enumerate(train_loader):
        img = sample['img'].cuda(device=opt.gpu)
        mask = sample['mask'].float().cuda(device=opt.gpu)
        n_iters += opt.batchSize
        # forward
        pred = model(img)

        optimizer.zero_grad()
        loss = criterionBCE(pred, mask)
        loss.backward()
        optimizer.step()

        train_losses.update(loss.data.item(), opt.batchSize)

        # score
        _iou, _f1 = pred2score(pred, mask)
        if _iou:
            train_iou.update(_iou, opt.batchSize)
            train_f1.update(_f1, opt.batchSize)

        if (i+1) % 5000 ==0 or i==len(train_loader)-1:
            save_log('Train: [{0}][{1}/{2}]\tLoss {loss_g.val:.4f} Loss_ave {loss_g.avg:.4f}'.format(
                epoch, i + 1, len(train_loader), loss_g=train_losses), "results/%s/out.txt" % (name))

for epoch in range(opt.epoch_start, opt.nEpochs):
    train_one_epoch(model, train_loader)
    torch.save(model.state_dict(), "results/%s/rf101_%03d.pth" % (name, epoch))