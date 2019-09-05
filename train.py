import gamegym as gym
from match import RandomAgent, Agent, SoftAgent, Evaluator

from algorithm.az import Nets, Planner, Trainer
#from algorithm.mctsbymcts import Nets, Planner, Trainer

args = {
    'batch_size': 64,
    'num_epochs': 30,
    'num_games': 999000,
    'num_train_steps': 120,
    'num_simulations': 50,
    'num_process': 7,
    'num_eval_process': 7,
    'concurrent_train': True,
    'bandit': 't', # u:UCB, t:Thompson
    'meta_bandit': 'u',
    'posterior': 't', # n:count, 't':Thompson
}

env = gym.make('TicTacToe')
#env = gym.make('Reversi')
#env = gym.make('AnimalShogi')
#env = gym.make('Go')

evaluator = Evaluator(env, args)

def evaluation(env, planner):
    # vs random
    agents = [Agent(planner), RandomAgent()]
    print('rand= ', evaluator.start(agents, True, 1000))
    # vs myself
    agents = [SoftAgent(planner), SoftAgent(planner)]
    print('self= ', evaluator.start(agents, False, 1000))

if env.game == 'TicTacToe':
    planner = Planner(Nets(env), args)

    s = env.State()
    s.plays('A1 C1 A2 C2')
    planner.inference(s, 20000, show=True)

    s = env.State()
    s.plays('B2 A2 A3 C1 B3')
    planner.inference(s, 20000, show=True)

    s = env.State()
    s.plays('B1 A3')
    planner.inference(s, 10000, show=True)


trainer = Trainer(env, args)
nets = trainer.run(callback=evaluation)

print(nets.inference(env.State()))
