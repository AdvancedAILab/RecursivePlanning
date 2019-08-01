  
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
        o = self.model.predict(state.feature())
        p = o['policy']
        ap_list = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])
        return ap_list[0][0]

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
    env, agents, n, process_id, num_process = args
    rewards = []
    for i in range(process_id, n, num_process):
        if i % 2 == 0:
            reward = do_match(env, [agents[0], agents[1]])
        else:
            reward = -do_match(env, [agents[1], agents[0]])

        rewards.append(reward)
    return rewards

def evaluate(env, agents, n, num_process=1):
    if num_process == 1:
        results = evaluate_process((env, agents, n, 0, 1))
    else:
        import multiprocessing as mp
        with mp.Pool(num_process) as p:
            results = p.map(evaluate_process, [(env, agents, n, i, num_process) for i in range(num_process)])
        results = sum(results, [])

    # gather results
    distribution = {}
    for reward in results:
        if reward not in distribution:
            distribution[reward] = 0
        distribution[reward] += 1
    return distribution
