
# Agent Class and Evaluation

import multiprocessing as mp
import numpy as np

class RandomAgent:
    def action(self, state):
        # random action
        return np.random.choice(state.legal_actions())

class Agent(RandomAgent):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def action(self, state):
        o = self.model.inference(state)
        p = o['policy']
        ap_list = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])
        return ap_list[0][0]

class SoftAgent(Agent):
    def action(self, state):
        o = self.model.inference(state)
        p = o['policy']
        mask = np.zeros_like(p)
        mask[state.legal_actions()] = 1
        p = (p + 1e-16) * mask
        return np.random.choice(np.arange(len(p)), p=p/p.sum())


def do_match(env, agents):
    state = env.State()
    turn = 0
    while not state.terminal():
        agent = agents[turn]
        action = agent.action(state)
        state.play(action)
        turn = 1 - turn
    reward = state.reward(subjective=False)
    return reward

def evaluate(env, agents, flip, n, process_id, num_process):
    rewards = []
    for i in range(process_id, n, num_process):
        if not flip or i % 2 == 0:
            reward = do_match(env, [agents[0], agents[1]])
        else:
            reward = -do_match(env, [agents[1], agents[0]])

        rewards.append(reward)
    return rewards


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.conns = None

    def run(self, conn):
        while True:
            args = conn.recv()
            if args is None:
               break
            agents = args[0]
            results = evaluate(self.env, agents, *args[1:])
            conn.send(results)

    def start(self, agents, flip, n):
        process = self.args['num_eval_process']
        if process == 1:
            results = evaluate(self.env, agents, flip, n, 0, 1)
        else:
            if self.conns is None:
                server_conns = []
                for i in range(process):
                    conn0, conn1 = mp.Pipe(duplex=True)
                    server_conns.append(conn1)
                    mp.Process(target=self.run, args=([conn0])).start()
                self.conns = server_conns

            for i, conn in enumerate(self.conns):
                conn.send((agents, flip, n, i, process))

            results = []
            for conn in self.conns:
                results.extend(conn.recv())

        # gather results
        distribution = {}
        for reward in results:
            if reward not in distribution:
                distribution[reward] = 0
            distribution[reward] += 1
        return dict(sorted(distribution.items(), reverse=True))