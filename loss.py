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

# class CrossEntropyLoss2d(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, ignore_label=255, reduction='mean'):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

#     def forward(self, output, target):
#         """
#         Forward pass
#         :param output: torch.tensor (NxC)
#         :param target: torch.tensor (N)
#         :return: scalar
#         """
#         return self.nll_loss(output, target)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight   = use_weight
        # if self.use_weight:
        #     self.weight = torch.FloatTensor(
        #         [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        #          0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        #          1.0865, 1.1529, 1.0507])
        #     print('CrossEntropy2d weights : {}'.format(self.weight))
        # else:
        #     self.weight = None


    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # Variable(torch.randn(2,10)
        if self.use_weight:
            print('target size {}'.format(target.shape))
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            self.weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = None


        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        # torch.FloatTensor([2.87, 13.19, 5.11, 37.98, 35.14, 30.9, 26.23, 40.24, 6.66, 32.07, 21.08, 28.14, 46.01, 10.35, 44.25, 44.9, 44.25, 47.87, 40.39])
        #weight = Variable(torch.FloatTensor([1, 1.49, 1.28, 1.62, 1.62, 1.62, 1.64, 1.62, 1.49, 1.62, 1.43, 1.62, 1.64, 1.43, 1.64, 1.64, 1.64, 1.64, 1.62]), requires_grad=False).cuda()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss


