from collections.abc import Iterable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(filters0, filters1, kernel_size,
                                stride=1, padding=kernel_size//2, bias=bias)
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class WideResidual(nn.Module):
    def __init__(self, filters, kernel_size, bn):
        super().__init__()
        self.conv1 = Conv(filters, filters, kernel_size, bn, not bn)
        self.conv2 = Conv(filters, filters, kernel_size, bn, not bn)

    def forward(self, h):
        return F.relu(h + self.conv2(F.relu(self.conv1(h))))

class BaseNet(nn.Module):
    def inference(self, *args):
        self.eval()
        with torch.no_grad():
            torch_args = [torch.FloatTensor(np.array(x)).unsqueeze(0) for x in args]
            outputs = self.forward(*torch_args)
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            return tuple([o.cpu().numpy()[0] for o in outputs])
        else:
            return outputs.cpu().numpy()[0]