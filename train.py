import gameenv as gym
from match import RandomAgent, Agent, evaluate

#from algorithm.az import Nets, Planner, Trainer
from algorithm.mctsbymcts import Nets, Planner, Trainer


args = {
    'batch_size': 16,
    'num_epochs': 30,
    'num_games': 9000,
    'num_train_steps': 30,
    'num_simulations': 50,
    'num_process': 3,
    'num_eval_process': 4,
}

env = gym.make('TicTacToe')
#env = gym.make('Reversi')

def vs_random(env, planner):
    agents = [Agent(planner), RandomAgent()]
    results = evaluate(env, agents, 1000, args['num_eval_process'])
    print(results)

planner = Planner(Nets(env))

s = env.State()
s.plays('A1 C1 A2 C2')
planner.predict(s, 20000, show=True)

s = env.State()
s.plays('B2 A2 A3 C1 B3')
planner.predict(s, 20000, show=True)

s = env.State()
s.plays('B1 A3')
planner.predict(s, 10000, show=True)

trainer = Trainer(env, args)
nets = trainer.run(callback=vs_random)

print(nets.predict(env.State()))

planner(env.State(), 20000, show=True)