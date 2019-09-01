  
# Agent Class

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

def evaluate_process(args):
    env, agents, flip, n, process_id, num_process = args
    rewards = []
    for i in range(process_id, n, num_process):
        if not flip or i % 2 == 0:
            reward = do_match(env, [agents[0], agents[1]])
        else:
            reward = -do_match(env, [agents[1], agents[0]])

        rewards.append(reward)
    return rewards

def evaluate(env, agents, flip, n, num_process=1):
    if num_process == 1:
        results = evaluate_process((env, agents, flip, n, 0, 1))
    else:
        import multiprocessing as mp
        with mp.Pool(num_process) as p:
            argss = [(env, agents, flip, n, i, num_process) for i in range(num_process)]
            results = p.map(evaluate_process, argss)
        results = sum(results, [])

    # gather results
    distribution = {}
    for reward in results:
        if reward not in distribution:
            distribution[reward] = 0
        distribution[reward] += 1
    return distribution
