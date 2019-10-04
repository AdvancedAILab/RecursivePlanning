
from .torch_module import ConvLSTMCell
from .board2d import Decoder

class DRC(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.num_layers = num_layers

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(ConvLSTMCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, num_repeats, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.shape[2:], x.shape[0])

        hs = [hidden[0][:,i] for i in range(self.d)]
        cs = [hidden[1][:,i] for i in range(self.d)]
        for _ in range(self.num_repeats):
            for i, block in enumerate(self.blocks):
                hs[i], cs[i] = block(x, (hs[i], cs[i]))

        return h[-1], (torch.stack(hs, dim=1), torch.stack(cs, dim=1))

    def init_hidden(self, input_size, batch_size):
        hs, cs = [], []
        for block in self.blocks:
            h, c = block.init_hidden(input_size, batch_size)
            hs.append(h)
            cs.append(c)

        return torch.stack(hs, dim=1), torch.stack(cs, dim=1)

class DRCEncoder(DRC):
    def __init__(self, env):
        num_layers = 3

        state = env.State()
        input_shape = state.feature().shape

        super().__init__(
            num_layers=num_layers, input_dim=input_shape[0], hidden_dim=32,
            kernel_size=input_shape[1:], bias=True
        )

class Nets(dict):
    def __init__(self, env):
        super().__init__({
            'encoder': DRCEncoder(env),
            'decoder': Decoder(env)
        })

    def __call__(self, x, num_repeats=1):
        encoded = self['encoder'](x)
        p, v = self['decoder'](encoded)
        return {'policy': p, 'value': v}

    def inference(self, state, num_repeats=1):
        x = state.feature()
        encoded = self['encoder'].inference(x)
        p, v = self['decoder'].inference(encoded)
        return {'policy': p, 'value': v}

class Planner:
    def __init__(self, nets, args):
        self.nets = nets

    def inference(self, state):
        return self.nets.inference(state, num_repeats=3)


if __name__ == '__main__':
    net = DRC(3, 3, 8, 16, (3, 3), True)
    x = torch.randn(5, 8, 4, 4)

    y, h = net(x, 1)

    print(y.size(), h[0].size(), h[1].size())