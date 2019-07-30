import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_module import *

class Encoder(nn.Module):
    def __init__(self, env):
        super().__init__()

        state = env.State()
        input_shape = state.feature().shape
        self.board_size = input_shape[1] * input_shape[2]

        self.conv0 = Conv(input_shape[0], 16, 3, bn=True)
        self.blocks = nn.ModuleList([WideResidual(16, 3, bn=True) for _ in range(4)])

    def forward(self, x):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = block(h)
        return h

class Decoder(nn.Module):
    def __init__(self, env, ):
        super().__init__()

        state = env.State()
        input_shape = state.feature().shape
        self.board_size = input_shape[1] * input_shape[2]

        self.conv_p = Conv(16, 2, 3, bn=True)
        self.fc_p = nn.Linear(self.board_size * 2, env.State().action_length())
        
        self.conv_v = Conv(16, 1, 3, bn=True)
        self.fc_v = nn.Linear(self.board_size * 1, 1)
    
    def forward(self, encoded_dep):
        h = encoded_dep
        h_p = F.relu(self.conv_p(h))
        h_v = F.relu(self.conv_v(h))
        h_p = self.fc_p(h_p).view(h_p.size(0), -1)
        h_v = self.fc_v(h_v)
        return F.softmax(h, dim=-1), torch.tanh(h_v)