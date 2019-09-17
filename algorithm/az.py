
# Alpha Zero

import time, copy
import multiprocessing as mp
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# domain dependent nets

from .board2d import Encoder, Decoder
from .bandit import *

# encoder   ... (domain dependent) feature -> dependent
# decoder   ... (domain dependent) encoded -> p, v

vloss = 4

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

    def inference(self, state):
        x = state.feature()
        encoded = self['encoder'].inference(x)
        p, v = self['decoder'].inference(encoded)
        return {'policy': p, 'value': v}

class Node:
    def __init__(self, state, outputs):
        self.p, self.v = outputs['policy'], outputs['value']
        self.q_sum, self.n = np.zeros_like(self.p), np.zeros_like(self.p)
        self.q_sum_all, self.n_all = 0, 0
        mask = np.zeros_like(self.p)
        mask[state.legal_actions()] = 1
        self.action_mask = (1 - mask) * 1e32

        self.p = (self.p + 1e-16) * mask
        self.p /= self.p.sum()

    def update(self, action, q_new):
        self.n[action] += 1
        self.q_sum[action] += q_new
        self.n_all += 1
        self.q_sum_all += q_new

        self.remove_vloss(action)

    def best(self):
        return int(np.argmax(self.n))

    def bandit(self, depth, method):
        p = self.p
        if depth == 0:
            p = 0.75 * p + 0.25 * np.random.dirichlet(np.ones_like(p) * 0.1)
            p /= p.sum()
        elif depth == -1:
            p = p + 0.1
            p /= p.sum()

        # apply bandit
        if method == 'u':
            action, info = pucb(self, p)
        elif method == 'r':
            action, info = ucbroot(self, p)
        else:
            action, info = pthompson(self, p)

        self.n[action] += vloss
        self.q_sum[action] += vloss * -1

        return action, info

    def remove_vloss(self, action):
        self.n[action] -= vloss
        self.q_sum[action] -= vloss * -1

class Planner:
    def __init__(self, nets, args):
        self.nets = nets
        self.args = args
        self.clear()

    def clear(self):
        self.node = {}
        self.store = {}

    def search(self, state, depth):
        if state.terminal():
            return state.reward(subjective=True)

        key = str(state)
        if key not in self.node:
            if key in self.store:
                self.extention += self.args['net_cache_extention']
                outputs = self.store[key]
            else:
                outputs = self.nets.inference(state)
                self.store[key] = outputs
            node = self.node[key] = Node(state, outputs)
            return node.v

        node = self.node[key]
        best_action, _ = node.bandit(depth, self.args['bandit'])

        state.play(best_action)
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)

        return q_new

    def inference(self, state, num_simulations, temperature=0, show=False):
        if show:
            print(state)
        start, prev_time = time.time(), 0
        self.node = {}
        self.extention = 0
        cnt = 0
        while cnt < num_simulations + self.extention:
            self.search(state.copy(), 0)
            cnt += 1

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

        if self.args['posterior'] == 'n':
            posterior = root.n / root.n.sum()
        else:
            posterior = pthompson_posterior(root, 4)
        posterior = (posterior / posterior.max()) ** (1 / (temperature + 1e-4))

        policy = posterior / posterior.sum()
        v = np.dot(policy, mean(root, 0.1))

        return {
            'policy': policy,
            'value': v,
        }

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def run(self, args):
        nets, _, st, n, process_id, num_process = args
        np.random.seed(process_id)
        episodes = []
        for g in range(st + process_id, st + n, num_process):
            print(g, '', end='', flush=True)
            episodes.append(self.generation(nets))
        return episodes

    def generation(self, nets, guide=''):
        record, ps, vs = [], [], []
        state = self.env.State()
        guide = state.str2path(guide)
        planner = Planner(nets, self.args)
        temperature = self.args['temperature']
        while not state.terminal():
            outputs = planner.inference(state, self.args['num_simulations'], temperature)
            policy = outputs['policy']
            if len(record) < len(guide):
                action = guide[len(record)]
            else:
                action = np.random.choice(np.arange(len(policy)), p=policy)
            state.play(action)
            record.append(action)
            ps.append(policy)
            vs.append(outputs['value'])
            temperature *= self.args['temp_decay']
        reward = state.reward(subjective=False)
        return record, reward, ps, vs

class Trainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.nets = None
        self.episodes = []
        self.reward_distribution = {}
        self.seed = int(self.args['seed'])
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def feed_episode(self, episode):
        # update stats
        self.episodes.append(episode)
        reward = episode[1]
        if reward not in self.reward_distribution:
            self.reward_distribution[reward] = 0
        self.reward_distribution[reward] += 1

    def generation_starter(self, nets, g):
        steps, process = self.args['num_train_steps'], self.args['num_process']
        if process == 1:
            episodes = Generator(self.env, self.args).run((nets, None, g, steps, 0, 1))
        else:
            with mp.Pool(process) as p:
                args_list = [(nets, None, g, steps, i, process) for i in range(process)]
                episodes = p.map(Generator(self.env, self.args).run, args_list)
            episodes = sum(episodes, [])
        for ep in episodes:
            self.feed_episode(ep)
        print('\nepisodes = %d' % (len(self.tree), len(self.episodes)))

    def train(self, gen, dice):
        #nets, params = Nets(self.env), []
        nets, params = copy.deepcopy(self.nets), []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for net in nets.values():
            net.train().to(device)
            params.extend(list(net.parameters()))

        optimizer = optim.SGD(params, lr=self.args['learning_rate'], weight_decay=1e-4, momentum=0.75)

        p_loss_sum, v_loss_sum = 0, 0
        max_datum = self.args['num_epochs'] * len(self.episodes)
        for dcnt in range(0, max_datum, self.args['batch_size']):
            if dcnt > 0 and self.stop_train:
                break

            ep_idx = dice.randint(len(self.episodes), size=self.args['batch_size'])
            x, p_target, v_target = zip(*[gen(self.episodes[idx], dice) for idx in ep_idx])

            x = torch.FloatTensor(np.array(x)).to(device).contiguous()
            p_target = torch.FloatTensor(np.array(p_target)).to(device).contiguous()
            v_target = torch.FloatTensor(np.array(v_target)).to(device).contiguous()

            o = nets(x)
            p_loss = torch.sum(p_target * torch.log(torch.clamp(p_target, 1e-12, 1) / torch.clamp(o['policy'], 1e-12, 1)))
            v_loss = torch.sum((v_target - o['value']) ** 2)

            p_loss_sum += p_loss.item()
            v_loss_sum += v_loss.item()

            optimizer.zero_grad()
            (p_loss + v_loss).backward()
            optimizer.step()

        print('p_loss %f v_loss %f' % (p_loss_sum / dcnt, v_loss_sum / dcnt))
        if not np.isnan(p_loss_sum):
            for net in nets.values():
                net.cpu()
            self.nets = nets

    def notime_planner(self, nets):
        return nets

    def gen_target(self, ep, dice):
        turn_idx = dice.randint(len(ep[0]))
        state = self.env.State()
        for a in ep[0][:turn_idx]:
            state.play(a)
        v = ep[1] if turn_idx % 2 == 0 else -ep[1]
        #v = ep[-1][turn_idx]
        return state.feature(), ep[2][turn_idx], [v]

    def run(self, callback=None):
        self.nets = Nets(self.env)
        print(self.nets.inference(self.env.State()))
        dice_train = np.random.RandomState(123)

        for g in range(0, self.args['num_games'], self.args['num_train_steps']):
            current_nets = copy.deepcopy(self.nets)

            if g > 0:
                # start training
                self.stop_train = False
                if self.args['concurrent_train']:
                    train_thread = threading.Thread(target=self.train, args=([self.gen_target, dice_train]))
                    train_thread.start()
                else:
                    self.train(self.gen_target, dice_train)

            if callback is not None:
                callback(self.env, self.notime_planner(current_nets))

            # episode generation
            self.generation_starter(self.nets, g)
            print('gen = ', dict(sorted(self.reward_distribution.items(), reverse=True)))

            if g > 0 and self.args['concurrent_train']:
                self.stop_train = True
                train_thread.join()

        return self.nets
