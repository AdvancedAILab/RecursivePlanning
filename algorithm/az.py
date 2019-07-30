  
# Alpha Zero

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# domain dependent nets

from .board2d import Encoder, Decoder

# encoder   ... (domain dependent) feature -> dependent
# decoder   ... (domain dependent) encoded -> p, v

class Node:
    def __init__(self, state, nets):
        self.feature = state.feature()
        self.encoded = nets['encoder'].predict(self.feature)
        self.p, self.v = nets['decoder'].predict(self.feature)
        self.q, self.n = np.zeros_like(self.p), np.zeros_like(self.p)
        self.n_all = 0
        self.action_mask = np.ones_like(self.p) * 2e32
        for a in state.legal_actions():
            self.action_mask[a] = 0

    def update(self, action, q_new):
        self.q[action] = (self.q[action] * self.n[action] + q_new) / (self.n[action] + 1)
        self.n[action] += 1

class Planner:
    def __init__(self, nets):
        self.nets = nets
        self.clear()

    def clear(self):
        self.node = {}

    def net(self, state):
        node = Node(state, self.nets)
        return node.p, node.v

    def search(self, state, depth):
        if state.terminal():
            return state.reward()

        key = str(state)
        if key not in self.node:
            self.node[key] = Node(state, self.nets)
            return self.node[key].v

        node = self.node[key]
        p = node.p
        if depth == 0:
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.1] * len(p))
        # pucb
        ucb = node.q + 2.0 * np.sqrt(node.n_all) * p / (node.n + 1) - node.action_mask

        best_action = np.argmax(ucb)
        state.play(best_action)
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature=0):
        for _ in range(num_simulations):
            self.search(state.copy(), 0)

        n = self.node[str(state)].n
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

class Trainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.episodes = []
        self.reward_distribution = {}

    def new_nets(self):
        return {
            'encoder': Encoder(self.env),
            'decoder': Decoder(self.env),
        }

    def generation(self, nets):
        record, p_targets = [], []
        state = self.env.State()
        planner = Planner(nets)
        temperature = 0.7
        print(state)
        while not state.terminal():
            p_target = planner.think(state, self.args['num_simulations'], temperature)
            action = np.random.choice(np.arange(len(p_target)), p=p_target)
            state.play(action)
            record.append(action)
            p_targets.append(p_target)
            temperature *= 0.8
        reward = state.reward(subjective=False)
        if reward not in self.reward_distribution:
            self.reward_distribution[reward] = 0
        self.reward_distribution[reward] += 1
        return record, reward, p_targets

    def train(self):
        def gen_target(ep):
            turn_idx = np.random.randint(len(ep[0]))
            state = self.env.State()
            for a in ep[0][:turn_idx]:
                state.play(a)
            v = ep[1]
            return state.feature(), ep[2][turn_idx], [v if turn_idx % 2 == 0 else -v]

        nets = self.new_nets()
        for net in nets.values():
            net.train()
        #optimizer = optim.SGD([net.parameters() for net in nets.values()], lr=1e-4, weight_decay=1e-4)
        optimizer = optim.SGD(nets['encoder'].parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(self.args['num_epochs']):
            p_loss_sum, v_loss_sum = 0, 0
            for i in range(0, len(self.episodes), self.args['batch_size']):
                ep_idx = np.random.randint(len(self.episodes), size=(self.args['batch_size']))
                x, p_target, v_target = zip(*[gen_target(self.episodes[idx]) for idx in ep_idx])
                x = torch.FloatTensor(np.array(x))
                p_target = torch.FloatTensor(np.array(p_target))
                v_target = torch.FloatTensor(np.array(v_target))

                encoded = self.nets['encoder'](x)
                p, v = self.nets['decoder'](encoded)
                p_loss = torch.sum(-p_target * torch.log(p))
                v_loss = torch.sum((v_target - v) ** 2)

                p_loss_sum += p_loss.item()
                v_loss_sum += v_loss.item()

                optimizer.zero_grad()
                (p_loss + v_loss).backward()
                optimizer.step()

        print('p_loss %f v_loss %f' % (p_loss_sum / len(self.episodes), v_loss_sum / len(self.episodes)))
        return nets

    def run(self):
        nets = self.new_nets()
        for g in range(self.args['num_games']):
            print(g, '', end='')
            self.episodes.append(self.generation(nets))

            if (g + 1) % self.args['num_train_steps'] == 0:
                print(self.reward_distribution)
                nets = self.train()
        return Planner(nets)