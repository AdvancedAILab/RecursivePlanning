  
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
        self.action_mask = np.ones_like(self.p) * 1e32
        for a in state.legal_actions():
            self.action_mask[a] = 0

    def update(self, action, q_new):
        self.n[action] += 1
        self.q_sum[action] += q_new
        self.n_all += 1
        self.q_sum_all += q_new

        self.remove_vloss(action)

    def best(self):
        return int(np.argmax(self.n))

    def bandit(self, depth):
        p = self.p
        if depth == 0:
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.1] * len(p))

        # pucb
        q_sum_all, n_all = self.q_sum_all + self.v / 2, self.n_all + 1
        q = (q_sum_all / n_all + self.q_sum) / (1 + self.n)
        ucb = q + 2.0 * np.sqrt(n_all) * p / (self.n + 1) - self.action_mask
        action = np.argmax(ucb)

        self.n[action] += vloss
        self.q_sum[action] += vloss * -1

        return action, ucb

    def remove_vloss(self, action):
        self.n[action] -= vloss
        self.q_sum[action] -= vloss * -1

class Planner:
    def __init__(self, nets):
        self.nets = nets
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
                self.bonus += 0.5
                outputs = self.store[key]
            else:
                outputs = self.nets.inference(state)
                self.store[key] = outputs
            node = self.node[key] = Node(state, outputs)
            return node.v

        node = self.node[key]
        best_action, _ = node.bandit(depth)

        state.play(best_action)
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)

        return q_new

    def inference(self, state, num_simulations, temperature=0, show=False):
        if show:
            print(state)
        start, prev_time = time.time(), 0
        self.node = {}
        self.bonus = 0
        cnt = 0
        while cnt < num_simulations + self.bonus:
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
        n = (root.n / np.max(root.n)) ** (1 / (temperature + 1e-8))
        n = np.maximum(n - root.action_mask, 0) # mask invalid actions

        return {
            'policy': n / n.sum(),
            'value': root.q_sum_all / root.n_all
        } 

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def run(self, args):
        nets, _, st, n, process_id, num_process = args
        episodes = []
        for g in range(st + process_id, st + n, num_process):
            print(g, '', end='', flush=True)
            episodes.append(self.generation(nets))
        return episodes

    def generation(self, nets, guide=[]):
        record, ps, vs = [], [], []
        state = self.env.State()
        planner = Planner(nets)
        temperature = 0.7
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
            temperature *= 0.8
        reward = state.reward(subjective=False)
        return record, reward, ps, vs

class Trainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.nets = None
        self.episodes = []
        self.reward_distribution = {}

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

    def train(self, gen):
        #nets, params = Nets(self.env), []
        nets, params = copy.deepcopy(self.nets), []
        for net in nets.values():
            net.train()
            params.extend(list(net.parameters()))

        optimizer = optim.SGD(params, lr=1e-3, weight_decay=1e-4, momentum=0.75)

        for _ in range(self.args['num_epochs']):
            if self.stop_train:
                break
            p_loss_sum, v_loss_sum = 0, 0
            for _ in range(0, len(self.episodes), self.args['batch_size']):
                ep_idx = np.random.randint(len(self.episodes), size=(self.args['batch_size']))
                x, p_target, v_target = zip(*[gen(self.episodes[idx]) for idx in ep_idx])
                x = torch.FloatTensor(np.array(x))
                p_target = torch.FloatTensor(np.array(p_target))
                v_target = torch.FloatTensor(np.array(v_target))

                o = nets(x)
                p_loss = torch.sum(p_target * torch.log(torch.clamp(p_target, 1e-12, 1) / torch.clamp(o['policy'], 1e-12, 1)))
                v_loss = torch.sum((v_target - o['value']) ** 2)

                p_loss_sum += p_loss.item()
                v_loss_sum += v_loss.item()

                optimizer.zero_grad()
                (p_loss + v_loss).backward()
                optimizer.step()

        print('p_loss %f v_loss %f' % (p_loss_sum / len(self.episodes), v_loss_sum / len(self.episodes)))
        self.nets = nets

    def notime_planner(self, nets):
        return nets

    def gen_target(self, ep):
        turn_idx = np.random.randint(len(ep[0]))
        state = self.env.State()
        for a in ep[0][:turn_idx]:
            state.play(a)
        v = ep[1] if turn_idx % 2 == 0 else -ep[1]
        return state.feature(), ep[2][turn_idx], [v]

    def run(self, callback=None):
        self.nets = Nets(self.env)
        print(self.nets.inference(self.env.State()))

        for g in range(0, self.args['num_games'], self.args['num_train_steps']):
            if callback is not None:
                callback(self.env, self.notime_planner(self.nets))

            if g > 0:
                # start training
                self.stop_train = False
                if self.args['concurrent_train']:
                    train_thread = threading.Thread(target=self.train, args=([self.gen_target]))
                    train_thread.start()
                else:
                    self.train(self.gen_target)

            # episode generation
            self.generation_starter(self.nets, g)
            print('gen = ', dict(sorted(self.reward_distribution.items(), reverse=True)))

            if g > 0 and self.args['concurrent_train']:
                self.stop_train = True
                train_thread.join()

        return self.nets
