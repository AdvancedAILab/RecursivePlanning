from collections.abc import Iterable

import numpy as np

import torch
torch.set_num_threads(1)

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

class ConvLSTMCell(nn.Module):
    # https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, input_size, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_dim, *input_size),
            torch.zeros(batch_size, self.hidden_dim, *input_size)
        )