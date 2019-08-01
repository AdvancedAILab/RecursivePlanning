import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_module import *

class Encoder(BaseNet):
    def __init__(self, env):
        super().__init__()

        state = env.State()
        input_shape = state.feature().shape
        self.board_size = input_shape[1] * input_shape[2]

        self.conv = Conv(input_shape[0], 16, 3, bn=True)
        self.blocks = nn.ModuleList([WideResidual(16, 3, bn=True) for _ in range(2)])

    def forward(self, x):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = block(h)
        return h

class Decoder(BaseNet):
    def __init__(self, env):
        super().__init__()

        state = env.State()
        input_shape = state.feature().shape
        self.board_size = input_shape[1] * input_shape[2]
        self.action_length = env.State().action_length()

        self.conv_p = Conv(16, 2, 1, bn=True)
        self.fc_p = nn.Linear(self.board_size * 2, self.action_length, bias=False)

        self.conv_v = Conv(16, 1, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 1, 1, bias=False)

    def forward(self, encoded):
        h = encoded

        h_p = F.relu(self.conv_p(h))
        h_p = self.fc_p(h_p.view(-1, self.board_size * 2))

        h_v = F.relu(self.conv_v(h))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 1))

        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

class PVNet(BaseNet):
    def __init__(self, env):
        super().__init__()
        self.encoder = Encoder(env)
        self.decoder = Decoder(env)

    def forward(self, x):
        return self.decoder(self.encoder(x))