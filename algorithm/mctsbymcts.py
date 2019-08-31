  
# Monte Carlo Tree Search by Monte Carlo Tree Search

import time, copy, pickle
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# domain dependent nets

from .az import Nets, Node, Planner
from .az import Generator as BaseGenerator
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
        sqn, k = self.book.size(state) ** 0.5, 8
        p = (o_book['policy'] * sqn + o_nets['policy'] * k) / (sqn + k)
        v = (o_book['value']  * sqn + o_nets['value']  * k) / (sqn + k)

        return {'policy': p, 'value': v}

class Generator(BaseGenerator):
    def run(self, args):
        nets, destination, st, n, process_id = args
        if isinstance(destination, mp.connection.Connection):
            # server-client mode
            conn = destination
            conn.send((process_id, [], None))
            while True:
                path = conn.recv()
                if path is None:
                    break
                episode = self.generation(nets, path)
                conn.send((process_id, path, episode))
        else:
            # single process mode
            master = destination
            for g in range(st, st + n):
                print(g, '', end='', flush=True)
                path = master.next_path()
                episode = self.generation(nets, path)
                master.feed_episode(path, episode)

class Trainer(BaseTrainer):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.tree = {}

    def next_path(self, base_path=[]):
        # decide next guide
        state = self.env.State()
        path = []

        for action in base_path:
            state.play(action)
            path.append(action)

        while str(state) in self.tree:
            node = self.tree[str(state)]
            action, _ = node.bandit(0)
            state.play(action)
            path.append(action)

        return path

    def cancel_path(self, path):
        # remove already added virtual losses
        state = self.env.State()
        for action in path:
            node = self.tree[str(state)]
            node.remove_vloss(action)

    def feed_episode(self, path, episode):
        super().feed_episode(episode)

        # feed episode to meta tree
        reward = episode[1] if len(episode[0]) % 2 == 0 else -episode[1]
        state = self.env.State()
        parents = []
        for d, action in enumerate(path):
            parents.append((self.tree[str(state)], action, episode[2][d], episode[3][d]))
            state.play(action)

        if len(path) < len(episode[0]):
            # sometimes guide path reaches terminal state
            p_leaf, v_leaf = episode[2][len(path)], episode[3][len(path)]
            key = str(state)
            if key not in self.tree:
                self.tree[key] = MetaNode(state, {'policy': p_leaf, 'value': v_leaf})
        else:
            v_leaf = reward * (1 if len(path) % 2 == 0 else -1)

        q_diff_sum = 0
        direction = -1
        for node, action, p, v in reversed(parents): # reversed order
            node.update(action, (v_leaf + q_diff_sum) * direction, reward * direction)

            v_old = node.v
            #alpha = 1 / node.n_all # mean
            alpha = 2 / (1 + node.n_all) # linear weight
            node.p = node.p * (1 - alpha) + p * alpha
            node.v = node.v * (1 - alpha) + v * alpha

            q_diff_sum += (node.v - v_old) * direction
            direction *= -1

    def next_path_delay(self, delay_info, num_conns):
        # decide next path with information of delayed paths
        path = None

        def check_path(path):
            path_bin = pickle.dumps(path)
            if path_bin in delay_info['path_sent']:
                delay_info['path'].append(path_bin)
                return None
            return path_bin

        while len(delay_info['path_ready']) > 0:
            path_bin = delay_info['path_ready'][0]
            delay_info['path_ready'].remove(path_bin)
            base_path = pickle.loads(path_bin)

            path = self.next_path(base_path)
            path_bin = check_path(path)
            if path_bin is not None:
                return path, path_bin

        # no delayed path was selected
        while len(delay_info['path']) < num_conns - 1:
            path = self.next_path()
            path_bin = check_path(path)
            if path_bin is not None:
                return path, path_bin

        return None, None

    def server(self, conns):
        # first requests to workers
        g = len(self.episodes)
        episodes = []
        waiting_conns = []
        delay_info = {
            'path': [],
            'path_ready': [],
            'path_sent': set(),
        }

        while len(conns) > 0:

            # receive results from generators
            conn_list = mp.connection.wait(conns)
            for conn in conn_list:
                _, path, episode = conn.recv()
                if episode is not None:
                    path_bin = pickle.dumps(path)
                    delay_info['path_sent'].remove(path_bin)
                    while path_bin in delay_info['path']:
                        # this path is ready to extract
                        delay_info['path'].remove(path_bin)
                        delay_info['path_ready'].append(path_bin)
                    episodes.append(episode)
                    self.feed_episode(path, episode)
            waiting_conns += conn_list

            # send next requests
            while len(waiting_conns) > 0:
                conn = waiting_conns[0]

                if len(episodes) + len(delay_info['path_sent']) < self.args['num_train_steps']:
                    path, path_bin = self.next_path_delay(delay_info, len(conns))
                    if path is None:
                        break

                    # send this path
                    delay_info['path_sent'].add(path_bin)
                    waiting_conns.remove(conn)
                    conn.send(path)
                    print(g, '', end='', flush=True)
                    g += 1
                else:
                    conn.send(None) # stop request
                    conns.remove(conn)
                    waiting_conns.remove(conn)

        # reset delayed paths to make tree consistent
        for path_bin in delay_info['path']:
            self.cancel_path(pickle.loads(path_bin))

        return episodes

    def generation_starter(self, nets, g):
        steps, process = self.args['num_train_steps'], self.args['num_process']
        if process == 1:
            Generator(self.env, self.args).run((nets, self, g, steps, 0))
        else:
            # make connection between server and worker
            server_conns = []
            for i in range(process):
                conn0, conn1 = mp.Pipe(duplex=True)
                server_conns.append(conn1)
                args = (nets, conn0, g, steps, i),
                p = mp.Process(target=Generator(self.env, self.args).run, args=args)
                p.start()
            self.server(server_conns)

        print('\nmeta tree size = %d episodes = %d' % (len(self.tree), len(self.episodes)))

    def notime_planner(self, nets):
        book = Book(self.tree)
        booknets = BookNets(book, nets)
        return booknets

    def gen_target(self, ep):
        turn_idx = np.random.randint(len(ep[0]))
        state = self.env.State()
        for a in ep[0][:turn_idx]:
            state.play(a)
        p = ep[2][turn_idx]
        v = ep[1] if turn_idx % 2 == 0 else -ep[1]

        # use result in meta-tree if found
        key = str(state)
        if key in self.tree:
            node = self.tree[key]
            if node.n_all > 0:
                p = node.p
                v = node.ro_sum_all / node.n_all

        return state.feature(), p, [v]
