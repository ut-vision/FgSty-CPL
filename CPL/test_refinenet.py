import argparse
import os, sys
import os.path as osp
import pprint
import random
import warnings
import cv2
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from hand import HandDataset
from refinenet import rf101
from utils import AverageMeter, prob2seg, save_log, pred2score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='refinenet')
parser.add_argument('--dataset', help='path to dataset', default="targets")
parser.add_argument('--batchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--test_mode', type=str, default="test", help='name of the test set')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--model_path', type=str, help='model path')
opt = parser.parse_args()

if opt.gpu >= 0:
    opt.cuda = True
    cudnn.deterministic = True
    cudnn.benchmark = False

print(opt)
root = "/path/to/your/root"
opt.dataset = opt.dataset.rstrip("/")

save_filename = opt.model_path.rstrip("/").split("/")[-1]
os.makedirs("results/%s" % (save_filename), exist_ok=True)
eval_models = [opt.model_path]

if opt.cuda:
    gpu_ids = [opt.gpu]
else:
    gpu_ids = []

if opt.dataset == "targets":
    test_sets = ["GTEA", "EDSH12", "EDSH1K", "UTG", "YHG"]
    test_db = opt.dataset
else:
    test_sets = [opt.dataset.split("/")[-1]]
    test_db = opt.dataset.split("/")[-1]
miou_dict = {"GTEA": 0, "EDSH12": 0, "EDSH1K": 0, "UTG": 0, "YHG": 0, "T_ave": 0}
mf1_dict = {"GTEA": 0, "EDSH12": 0, "EDSH1K": 0, "UTG": 0, "YHG": 0, "T_ave": 0}
model_name = opt.model_path.split("/")[-1]
model_id = int(model_name.split("_")[1]) if type(model_name.split("_")[1])==int else model_name
name = "_".join(["test_model", model_name, "dataset", test_db])
save_log("test model: {}".format(model_name), "results/%s/test.txt" % (save_filename))
ave_mIoU = []
ave_mf1 = []    
for dataset in test_sets:
    testset = HandDataset(os.path.join(root, dataset), opt.test_mode)
    test_loader = DataLoader(dataset=testset, batch_size=opt.batchSize, shuffle=True)

    model = rf101(opt.output_nc, pretrained=False, use_dropout=True).cuda(device=opt.gpu)
    #print(model)

    losses = AverageMeter()
    iou = AverageMeter()
    f1 = AverageMeter()

    model.load_state_dict(torch.load(opt.model_path, map_location="cuda:%d"%opt.gpu))
    model.eval()
    for i, sample in enumerate(test_loader):
        img = sample['img']
        mask = sample['mask'].float()
        filename = testset.filenames[i]
        if opt.cuda:
            img = img.cuda(device=opt.gpu)
            mask = mask.cuda(device=opt.gpu)
        
        # forward
        pred = model(img)
        if isinstance(pred, tuple):
            pred = pred[0]

        # score
        _iou, _f1 = pred2score(pred, mask)
        if _iou:
            iou.update(_iou, opt.batchSize)
            f1.update(_f1, opt.batchSize)  
    save_log('Test on {}:  m_size [{}]\tmIoU {iou.avg:.4f}, mF1 {f1.avg:.4f}'.format(dataset, len(testset), iou=iou, f1=f1), "results/%s/test.txt" % (save_filename))
    if opt.dataset == "targets":
        miou_dict[dataset] = iou.avg
        mf1_dict[dataset] = f1.avg
    if dataset != "EGTEA":
        ave_mIoU.append(iou.avg)
        ave_mf1.append(f1.avg)
mIoU_cross = np.mean(ave_mIoU)
save_log("ave mIoU {:.4f}".format(mIoU_cross), "results/%s/test.txt" % (save_filename))
save_log("ave mF1 {:.4f}".format(np.mean(ave_mf1)), "results/%s/test.txt" % (save_filename))
if opt.dataset == "targets":
    miou_dict["T_ave"] = np.mean(ave_mIoU)
    mf1_dict["T_ave"] = np.mean(ave_mf1)

    save_log(opt.model_path, "results/%s/test.txt" % (save_filename))
    save_log("mIoU\n|GTEA|EDSH2|EDSHK|UTG|YHG|T_ave|\n|:---:|:---:|:---:|:---:|:---:|:---:|", "results/%s/test.txt" % (save_filename))
    save_log("|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|({})".format(
        miou_dict["GTEA"]*100,
        miou_dict["EDSH12"]*100,
        miou_dict["EDSH1K"]*100,
        miou_dict["UTG"]*100,
        miou_dict["YHG"]*100,
        miou_dict["T_ave"]*100,
        model_name,
    ), "results/%s/test.txt" % (save_filename))
    save_log("mF1\n|GTEA|EDSH2|EDSHK|UTG|YHG|T_ave|\n|:---:|:---:|:---:|:---:|:---:|:---:|", "results/%s/test.txt" % (save_filename))
    save_log("|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|({})".format(
        mf1_dict["GTEA"]*100,
        mf1_dict["EDSH12"]*100,
        mf1_dict["EDSH1K"]*100,
        mf1_dict["UTG"]*100,
        mf1_dict["YHG"]*100,
        mf1_dict["T_ave"]*100,
        model_name,
    ), "results/%s/test.txt" % (save_filename))
