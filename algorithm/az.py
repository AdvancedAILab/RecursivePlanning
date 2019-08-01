  
# Alpha Zero

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# domain dependent nets

from .board2d import *

# encoder   ... (domain dependent) feature -> dependent
# decoder   ... (domain dependent) encoded -> p, v

class Nets(dict):
    def __init__(self, env):
        super().__init__({
            'encoder': Encoder(env),
            'decoder': Decoder(env)
        })

    def __call__(self, x):
        encoded = self['encoder'](x)
        p, v = self['decoder'](encoded)
        return {'policy': p, 'value': v}

    def predict(self, x):
        encoded = self['encoder'].predict(x)
        p, v = self['decoder'].predict(encoded)
        return {'policy': p, 'value': v}

class Node:
    def __init__(self, state, nets):
        o = nets.predict(state.feature())
        self.p, self.v = o['policy'], o['value']
        self.q_sum, self.n = np.zeros_like(self.p), np.zeros_like(self.p)
        self.n_all = 1
        self.q_sum_all = self.v / 2
        self.action_mask = np.ones_like(self.p) * 1e32
        for a in state.legal_actions():
            self.action_mask[a] = 0

    def update(self, action, q_new):
        self.n[action] += 1
        self.q_sum[action] += q_new
        self.n_all += 1
        self.q_sum_all += q_new

    def best(self):
        return int(np.argmax(self.n))

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
            return state.reward(subjective=True)

        key = str(state)
        if key not in self.node:
            node = self.node[key] = Node(state, self.nets)
            return node.v

        node = self.node[key]
        p = node.p
        if depth == 0:
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.1] * len(p))
        # pucb
        q = (node.q_sum_all / node.n_all + node.q_sum) / (1 + node.n)
        ucb = q + 2.0 * np.sqrt(node.n_all) * p / (node.n + 1) - node.action_mask

        best_action = np.argmax(ucb)
        state.play(best_action)
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)

        return q_new

    def __call__(self, state, num_simulations, temperature=0, show=False):
        if show:
            print(state)
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(state.copy(), 0)

            # show status every 1 second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root = self.node[str(state)]
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d'
                          % (tmp_time, state.action2str(root.best()),
                             root.q_sum[root.best()] / root.n[root.best()],
                             root.n[root.best()], root.n_all))

        root = self.node[str(state)]
        n = (root.n / np.max(root.n)) ** (1 / (temperature + 1e-8))
        n = np.maximum(n - root.action_mask, 0) # mask invalid actions
        return n / n.sum()

class Trainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.episodes = []
        self.reward_distribution = {}

    def generation(self, nets):
        record, p_targets = [], []
        state = self.env.State()
        planner = Planner(nets)
        temperature = 0.7
        while not state.terminal():
            p_target = planner(state, self.args['num_simulations'], temperature)
            action = np.random.choice(np.arange(len(p_target)), p=p_target)
            state.play(action)
            record.append(action)
            p_targets.append(p_target)
            temperature *= 0.8
        reward = state.reward(subjective=False)
        return record, reward, p_targets

    def generation_process(self, args):
        nets, st, n, process_id, num_process = args
        episodes = []
        for g in range(st + process_id, st + n, num_process):
            print(g, '', end='', flush=True)
            episodes.append(self.generation(nets))
        return episodes

    def train(self):
        def gen_target(ep):
            turn_idx = np.random.randint(len(ep[0]))
            state = self.env.State()
            for a in ep[0][:turn_idx]:
                state.play(a)
            v = ep[1]
            return state.feature(), ep[2][turn_idx], [v if turn_idx % 2 == 0 else -v]

        nets, params = Nets(self.env), []
        for net in nets.values():
            net.train()
            params.extend(list(net.parameters()))
        optimizer = optim.SGD(params, lr=1e-3, weight_decay=1e-4, momentum=0.75)
        for _ in range(self.args['num_epochs']):
            p_loss_sum, v_loss_sum = 0, 0
            for _ in range(0, len(self.episodes), self.args['batch_size']):
                ep_idx = np.random.randint(len(self.episodes), size=(self.args['batch_size']))
                x, p_target, v_target = zip(*[gen_target(self.episodes[idx]) for idx in ep_idx])
                x = torch.FloatTensor(np.array(x))
                p_target = torch.FloatTensor(np.array(p_target))
                v_target = torch.FloatTensor(np.array(v_target))

                o = nets(x)
                p_loss = torch.sum(-p_target * torch.log(o['policy']))
                v_loss = torch.sum((v_target - o['value']) ** 2)

                p_loss_sum += p_loss.item()
                v_loss_sum += v_loss.item()

                optimizer.zero_grad()
                (p_loss + v_loss).backward()
                optimizer.step()

        print('p_loss %f v_loss %f' % (p_loss_sum / len(self.episodes), v_loss_sum / len(self.episodes)))
        return nets

    def run(self, callback=None):
        nets = Nets(self.env)
        print(nets.predict(self.env.State().feature()))
        if callback is not None:
            callback(self.env, nets)

        steps, process = self.args['num_train_steps'], self.args['num_process']
        for g in range(0, self.args['num_games'], steps):
            if process == 1:
                episodes = self.generation_process((nets, g, steps, 0, 1))
            else:
                import multiprocessing as mp
                with mp.Pool(process) as p:
                    episodes = p.map(self.generation_process, [(nets, g, steps, i, process) for i in range(process)])
                episodes = sum(episodes, [])
            
            for _, reward, _ in episodes:
                if reward not in self.reward_distribution:
                    self.reward_distribution[reward] = 0
                self.reward_distribution[reward] += 1
            self.episodes.extend(episodes)

            print(self.reward_distribution)
            nets = self.train()
            if callback is not None:
                callback(self.env, nets)

        return nets