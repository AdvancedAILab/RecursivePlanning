
# Base Implementation of Training Process with Extensible Planning

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


class Generator:
    def __init__(self, env, planner, args):
        self.env = env
        self.planner = planner
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
        planner = self.planner(nets, self.args)
        temperature = self.args['temperature']
        while not state.terminal():
            outputs = planner.inference(state, temperature=temperature)
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
        self.seed = self.args['seed']
        if self.seed is not None:
            self.seed = int(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def feed_episode(self, episode):
        # update stats
        self.episodes.append(episode)
        reward = episode[1]
        if reward not in self.reward_distribution:
            self.reward_distribution[reward] = 0
        self.reward_distribution[reward] += 1

    def generation_starter(self, nets, planner, g):
        steps, process = self.args['num_train_steps'], self.args['num_process']
        if process == 1:
            episodes = Generator(self.env, planner, self.args).run((nets, None, g, steps, 0, 1))
        else:
            with mp.Pool(process) as p:
                args_list = [(nets, None, g, steps, i, process) for i in range(process)]
                episodes = p.map(Generator(self.env, planner, self.args).run, args_list)
            episodes = sum(episodes, [])
        for ep in episodes:
            self.feed_episode(ep)
        print('\nepisodes = %d' % len(self.episodes))

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

    def gen_target(self, ep, dice):
        turn_idx = dice.randint(len(ep[0]))
        state = self.env.State()
        for a in ep[0][:turn_idx]:
            state.play(a)
        v = ep[1] if turn_idx % 2 == 0 else -ep[1]
        #v = ep[-1][turn_idx]
        return state.feature(), ep[2][turn_idx], [v]

    def run(self, algoset, callback=None):
        nets, planner, instant_planner = algoset['nets'], algoset['planner'], algoset['instant_planner']
        self.nets = nets(self.env)
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
                callback(self.env, current_nets, 'net')
                if 'notime_planner' in dir(self):
                    callback(self.env, instant_planner(current_nets), 'n+t')

            # episode generation
            self.generation_starter(self.nets, planner, g)
            print('gen = ', dict(sorted(self.reward_distribution.items(), reverse=True)))

            if g > 0 and self.args['concurrent_train']:
                self.stop_train = True
                train_thread.join()

        return self.nets
