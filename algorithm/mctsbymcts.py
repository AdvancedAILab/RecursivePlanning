  
# Monte Carlo Tree Search by Monte Carlo Tree Search

import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# domain dependent nets

from .az import Nets, Node, Planner
from .az import Trainer as BaseTrainer

class MetaNode(Node):
    def __init__(self, state, outputs):
        super().__init__(state, outputs)
        self.ro_sum = np.zeros_like(self.p)
        self.ro_sum_all = 0

    def update(self, action, q_new, ro_new):
        super().update(action, q_new)
        self.ro_sum[action] += ro_new
        self.ro_sum_all += ro_new

class Book:
    def __init__(self, nodes):
        self.node = nodes

    def inference(self, state):
        key = str(state)
        if key in self.node:
            p, v = self.node[key].p, self.node[key].v
        else:
            al = state.action_length()
            p, v = np.ones((al)) / al, 0

        return {'policy': p, 'value': v}

    def size(self, state):
        key = str(state)
        return self.node[key].n_all if key in self.node else 0

class BookNets:
    def __init__(self, book, nets):
        self.book = book
        self.nets = nets

    def inference(self, state):
        o_book = self.book.inference(state)
        o_nets = self.nets.inference(state)
        # ratio; sqrt(n) : k
        sqn, k = self.book.size(state), 8
        p = (o_book['policy'] * sqn + o_nets['policy'] * k) / (sqn + k)
        v = (o_book['value']  * sqn + o_nets['value']  * k) / (sqn + k)

        return {'policy': p, 'value': v}

class Trainer(BaseTrainer):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.tree = {}

    def next_path(self):
        # decide next guide
        path = []
        state = self.env.State()

        while str(state) in self.tree:
            node = self.tree[str(state)]
            a, _ = node.bandit(0)
            state.play(a)
            path.append(a)

        return path

    def feed_episode(self, path, episode):
        alpha = 0.5

        state = self.env.State()
        parents = []
        for d, action in enumerate(path):
            p, v = episode[2][d], episode[3][d]
            node = self.tree[str(state)]

            diff = (v - node.v) * alpha
            direction = -1
            for nd, a in reversed(parents):
                nd.q_sum[a] += diff * direction
                nd.q_sum_all += diff * direction
                direction *= -1

            node.p = p * alpha + node.p * (1 - alpha)
            node.v = v * alpha + node.v * (1 - alpha)

            state.play(action)
            parents.append((node, action))

        reward = episode[1] if len(episode[0]) % 2 == 0 else -episode[1]
        if len(path) < len(episode[0]):
            # sometimes guide path reaches terminal state
            p, v = episode[2][len(path)], episode[3][len(path)]

            key = str(state) 
            if key not in self.tree:
                self.tree[key] = MetaNode(state, {'policy': p, 'value': v})
        else:
            v = reward

        direction = -1
        for nd, a in reversed(parents):
            nd.update(a, v * direction, reward * direction)
            direction *= -1

    def generation_process_solo(self, args):
        nets, _, st, n, process_id, num_process = args
        episodes = []
        for g in range(st + process_id, st + n, num_process):
            print(g, '', end='', flush=True)
            path = self.next_path()
            episode = self.generation(nets, path)
            self.feed_episode(path, episode)
            episodes.append(episode)
        return episodes

    def generation_process_multi(self, nets, conn, st, n, process_id, num_process):
        for g in range(st + process_id, st + n, num_process):
            print(g, '', end='', flush=True)
            conn.send(True)
            path = conn.recv()
            episode = self.generation(nets, path)
            conn.send((process_id, path, episode))
        conn.send(False) # finished flag

    def server(self, conns):
        # first requests to workers
        for conn in conns:
            if conn.recv():
                conn.send(self.next_path())

        episodes = []
        while len(conns) > 0:
            conn_list = mp.connection.wait(conns)
            for conn in conn_list:
                _, path, episode = conn.recv()
                episodes.append(episode)
                self.feed_episode(path, episode)
                if conn.recv():
                    conn.send(self.next_path())
                else:
                    conns.remove(conn)

        return episodes

    def generation_starter(self, nets, args, g):
        steps, process = self.args['num_train_steps'], self.args['num_process']
        if process == 1:
            episodes = self.generation_process_solo((nets, None, g, steps, 0, 1))
        else:
            # make connection between server and worker
            server_conns = []
            for i in range(process):
                conn0, conn1 = mp.Pipe(duplex=True)
                server_conns.append(conn1)
                p = mp.Process(target =self.generation_process_multi, args=(nets, conn0, g, steps, i, process))
                p.start()
            episodes = self.server(server_conns)

        print('meta tree size = %d' % len(self.tree))
        print('episodes = %d' % len(episodes))
        return episodes

    def notime_planner(self, nets):
        book = Book(self.tree)
        booknets = BookNets(book, nets)
        return booknets
