import argparse

import gamegym as gym
from match import RandomAgent, PerfectAgent, Agent, SoftAgent, Evaluator

default_args = {
    # algorithm
    'algo': 'MCTSbyMCTS',

    # environment
    'env': 'TicTacToe',

    # system
    'num_games': 20000,
    'num_train_steps': 100,
    'num_process': 1,
    'concurrent_train': False,
    'bandit': 'u', # u:UCB, t:Thompson
    'meta_bandit': 'u', # u:UCB, t:Thompson
    'posterior': 'n', # n:count, 't':Thompson
    'meta_p_randomization': True,

    # fitting neural nets
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3,

    # episode generation
    'num_simulations': 100,
    'net_cache_extention': 0,
    'temperature': 0.8,
    'temp_decay': 0.8,

    # experimental settings
    'seed': None,
}

parser = argparse.ArgumentParser()
for k, v in default_args.items():
    t = None if v is None or isinstance(v, bool) else type(v)
    parser.add_argument('--' + k, default=v, type=t)

args = vars(parser.parse_args())
print(args)

env = gym.make(args['env'])
evaluator = Evaluator(env, args)

def evaluation(env, planner, name):
    # vs random
    agents = [Agent(planner), RandomAgent()]
    print('%s:rand= %s' % (name, evaluator.start(agents, True, 1000)))
    # vs perfect
    if env.game == 'TicTacToe':
        agents = [Agent(planner), PerfectAgent()]
        print('%s:perf= %s' % (name, evaluator.start(agents, True, 1000)))
    # vs myself
    agents = [SoftAgent(planner), SoftAgent(planner)]
    print('%s:self= %s' % (name, evaluator.start(agents, False, 1000)))

if args['algo'] == 'AlphaZero':
    from algorithm.az import Nets, Planner, Trainer
elif args['algo'] == 'MCTSbyMCTS':
    from algorithm.mctsbymcts import Nets, Planner, Trainer

if env.game == 'TicTacToe':
    planner = Planner(Nets(env), args)

    s = env.State()
    s.plays('A1 C1 A2 C2')
    planner.inference(s, num_simulations=20000, show=True)

    s = env.State()
    s.plays('B2 A2 A3 C1 B3')
    planner.inference(s, num_simulations=20000, show=True)

    s = env.State()
    s.plays('B1 A3')
    planner.inference(s, num_simulations=10000, show=True)

trainer = Trainer(env, args)
nets = trainer.run(callback=evaluation)

print(nets.inference(env.State()))
