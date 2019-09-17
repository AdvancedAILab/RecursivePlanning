 
# Monte Carlo Tree Search by Monte Carlo Tree Search

import time, copy
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
            np.random.seed(process_id)
            conn = destination
            while True:
                nets = conn.recv()
                if nets is None:
                    break
                conn.send((process_id, '', None))
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
        self.conns = None

    def next_path(self, base_path=''):
        # decide next guide
        state = self.env.State()
        state.plays(base_path)

        while str(state) in self.tree:
            node = self.tree[str(state)]
            depth = 0 if self.args['meta_p_randomization'] else -1
            action, _ = node.bandit(depth, self.args['meta_bandit'])
            state.play(action)

        return state.record_string()

    def cancel_path(self, path):
        # remove already added virtual losses
        state = self.env.State()
        for action in state.str2path(path):
            node = self.tree[str(state)]
            node.remove_vloss(action)
            state.play(action)

    def feed_episode(self, path, episode):
        super().feed_episode(episode)

        # feed episode to meta tree
        reward = episode[1] if len(episode[0]) % 2 == 0 else -episode[1]
        state = self.env.State()
        path = state.str2path(path)
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

    def next_path_delay(self, status, num_conns):
        # decide next path with information of delayed paths
        def check_path(path):
            if path in status['sent']:
                status['delayed'].append(path)
                return False
            return True

        while len(status['ready']) > 0:
            base_path = status['ready'].pop(0)
            path = self.next_path(base_path)
            if check_path(path):
                return path

        # no delayed path was selected
        while len(status['delayed']) < num_conns - 1:
            path = self.next_path()
            if check_path(path):
                return path

        return None

    def server(self):
        # first requests to workers
        g = len(self.episodes)
        num_episodes = 0
        waiting_conns = []
        status = {
            'sent': set(), 'delayed': [], 'ready': []
        }
        for conn in self.conns:
            conn.send(self.nets)

        while num_episodes < self.args['num_train_steps']:
            # receive results from generators
            conn_list = mp.connection.wait(self.conns)
            for conn in conn_list:
                _, path, episode = conn.recv()
                if episode is not None:
                    status['sent'].remove(path)
                    while path in status['delayed']:
                        # this path is ready to extract
                        status['delayed'].remove(path)
                        status['ready'].append(path)
                    self.feed_episode(path, episode)
                    num_episodes += 1
            waiting_conns += conn_list

            # send next requests
            while len(waiting_conns) > 0:
                if num_episodes + len(status['sent']) < self.args['num_train_steps']:
                    path = self.next_path_delay(status, len(self.conns))
                    if path is None:
                        break

                    # send this path
                    status['sent'].add(path)
                    conn = waiting_conns.pop(0)
                    conn.send(path)
                    print(g, '', end='', flush=True)
                    g += 1
                else:
                    break

        for conn in self.conns:
            conn.send(None) # stop request

        # reset delayed paths to make tree consistent
        for path in status['delayed']:
            self.cancel_path(path)

    def generation_starter(self, nets, g):
        steps, process = self.args['num_train_steps'], self.args['num_process']
        if process == 1:
            Generator(self.env, self.args).run((nets, self, g, steps, 0))
        else:
            # make connection between server and worker
            if self.conns is None:
                server_conns = []
                for i in range(process):
                    conn0, conn1 = mp.Pipe(duplex=True)
                    server_conns.append(conn1)
                    args = (None, conn0, None, None, i),
                    mp.Process(target=Generator(self.env, self.args).run, args=args).start()
                self.conns = server_conns
            self.server()

        print('\nmeta tree size = %d episodes = %d' % (len(self.tree), len(self.episodes)))

    def notime_planner(self, nets):
        book = Book(self.tree)
        booknets = BookNets(book, nets)
        #return booknets
        return nets

    def gen_target(self, ep, dice):
        turn_idx = dice.randint(len(ep[0]))
        state = self.env.State()
        for a in ep[0][:turn_idx]:
            state.play(a)
        p = ep[2][turn_idx]
        v = ep[1] if turn_idx % 2 == 0 else -ep[1]
        #v = ep[-1][turn_idx]

        # use result in meta-tree if found
        key = str(state)
        if key in self.tree:
            node = self.tree[key]
            if node.n_all > 0:
                p = node.p
                v = node.ro_sum_all / node.n_all
                # v = node.v

        return state.feature(), p, [v]
