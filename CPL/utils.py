import numpy as np
import torch
import torch.nn.functional as F
import cv2, csv

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_log(log, output_file, isPrint=True):
    with open(output_file, "a") as f:
        f.write(log + "\n")
        if isPrint:
            print(log)

def prob2seg(prob):
    """
    convert probability map to 0/255 segmentation mask
    prob: probability map [0-255]
    """
    # smooth and thresholding
    prob = cv2.GaussianBlur(prob, (5, 5), 0)
    ret, mask = cv2.threshold(prob,75,1,cv2.THRESH_BINARY) # would remove the single-channel dimension
    
    # remove holes and spots
    kernel = np.ones((5,5),np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    # filter out small area
    contours, hierarchy = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_ratio = 0.002
    max_ratio = 0.2
    area_img = prob.shape[0] * prob.shape[1]
    mask_close = mask_close * 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > area_img*min_ratio and area < area_img*max_ratio:
            cv2.drawContours(mask_close, [contours[i]], -1, 1, -1)

    return mask_close * 255

def pred2score(pred, mask, out_list=False):
    # compute IoU
    scores = {"miou": [], "mf1": []}
    scores_list = {"miou": [], "mf1": [], "union": [], "intersection": []}
    for i in range(pred.size(0)):
        # compute segmentation map from probability map
        pred_numpy = np.clip(np.transpose(pred.data[i].cpu().numpy(), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        mask_numpy = np.clip(np.transpose(mask.data[i].cpu().numpy(), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        union = np.logical_or(mask_numpy>128, pred_numpy>128)
        intersection = np.logical_and(mask_numpy>128, pred_numpy>128)
        M = np.count_nonzero(mask_numpy>128)
        P = np.count_nonzero(pred_numpy>128)
        if np.count_nonzero(union) > 0:
            iou = np.count_nonzero(intersection)*1.0/np.count_nonzero(union)
            recall = np.count_nonzero(intersection)*1.0/(M+0.00001)
            precision = np.count_nonzero(intersection)*1.0/(P+0.00001)
            f1 = 2 * recall * precision / (recall + precision + 0.00001)
            scores_list["miou"].append(iou)
            scores_list["mf1"].append(f1)
            scores_list["union"].append(union)
            scores_list["intersection"].append(intersection)
            scores["miou"].append(iou)
            scores["mf1"].append(f1)
        else:
            scores_list["miou"].append(0)
            scores_list["mf1"].append(0)
            scores_list["union"].append(np.zeros_like(mask_numpy))
            scores_list["intersection"].append(np.zeros_like(mask_numpy))

    if out_list:
        return scores_list
    else:
        return np.mean(scores["miou"]), np.mean(scores["mf1"])