import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):

    def __init__(self, num_classes=8):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.classes = 3
        self.ignore_index = None
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc



class JaccardScore(nn.Module):

    def __init__(self):
        super(JaccardScore, self).__init__()
    
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = torch.logical_and(y_true, y_pred)
        union = torch.logical_or(y_true, y_pred)
        iou_score = torch.sum(intersection) / torch.sum(union)  
        return iou_score