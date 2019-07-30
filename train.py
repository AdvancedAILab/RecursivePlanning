import modules.GameImplementation.tictactoe as env
from algorithm.az import Planner, Trainer

args = {
    'batch_size': 32,
    'num_epochs': 30,
    'num_games': 150,
    'num_train_steps': 30,
    'num_simulations': 50,
}

trainer = Trainer(env, args)
planner = trainer.run()

planner.think(env.State(), 100000, show=True)