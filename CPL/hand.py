import os
import os.path as osp
import sys
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HandDataset(Dataset):
    def __init__(self, dataset, phase, PL_path=None, opt=""):
        super(HandDataset, self).__init__()
        self.imgs = []
        self.masks = []
        self.opt = opt
        if "train" in phase or "all" in phase:
            imageset = "%s/train" % (dataset)
            maskset = "%s/trainannot" % (dataset)
        elif "test" in phase:
            imageset = "%s/test" % (dataset)
            maskset = "%s/testannot" % (dataset)
        else:
            raise NotImplementedError()
        img_filenames = sorted([k for k in os.listdir(imageset) if (".png" in k or ".jpg" in k)])
        self.imgs.extend([imageset + '/' + filename for filename in img_filenames])
        self.filenames = sorted([k for k in os.listdir(maskset) if (".png" in k or ".jpg" in k)])
        self.masks.extend([maskset + '/' + filename for filename in self.filenames])
        if "all" in phase:
            imageset = "%s/test" % (dataset)
            maskset = "%s/testannot" % (dataset)
            img_filenames = sorted([k for k in os.listdir(imageset) if (".png" in k or ".jpg" in k)])
            self.filenames = sorted([k for k in os.listdir(maskset) if (".png" in k or ".jpg" in k)])
            self.imgs.extend([imageset + '/' + filename for filename in img_filenames])
            self.masks.extend([maskset + '/' + filename for filename in self.filenames])
        img_sample = cv2.imread(self.imgs[0])
        self.img_size = (img_sample.shape[1], img_sample.shape[0])
        transform_list = [transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])]
        self.load_size = (256, 256)
        self.transform = transforms.Compose(transform_list)
        print("len", len(self.imgs))
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        imgname = self.imgs[index]
        filename = imgname.split("/")[-1]
        img = cv2.imread(imgname)
        img = cv2.resize(img, self.load_size, interpolation = cv2.INTER_CUBIC)
        maskname = self.masks[index]
        mask = cv2.imread(maskname, 0)
        mask = cv2.resize(mask, self.load_size, interpolation = cv2.INTER_NEAREST)
        img_tensor = self.transform(img)
        _, mask_thresh = cv2.threshold(np.asarray(mask), 0, 1, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(mask_thresh)
        mask_tensor = mask_tensor.unsqueeze(0)

        sample = {'img': img_tensor, 'mask': mask_tensor, 'index': index, 'path': imgname}
        return sample
