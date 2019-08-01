import gameenv as gym
from algorithm.az import Nets, Planner, Trainer
from match import RandomAgent, Agent, evaluate

args = {
    'batch_size': 32,
    'num_epochs': 30,
    'num_games': 900,
    'num_train_steps': 30,
    'num_simulations': 50,
    'num_process': 3,
}

env = gym.make('TicTacToe')
#env = gym.make('Reversi')

def vs_random(env, planner):
    results = evaluate(env, [Agent(planner), RandomAgent()], 1000, args['num_process'])
    print(results)

planner = Planner(Nets(env))

s = env.State()
s.plays('A1 C1 A2 C2')
planner(s, 20000, show=True)

s = env.State()
s.plays('B2 A2 A3 C1 B3')
planner(s, 20000, show=True)

s = env.State()
s.plays('B1 A3')
planner(s, 20000, show=True)

trainer = Trainer(env, args)
nets = trainer.run(callback=vs_random)

print(nets.predict(env.State().feature()))

planner(env.State(), 20000, show=True)