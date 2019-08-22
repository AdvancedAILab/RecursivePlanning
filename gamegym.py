import modules.GameImplementation.games as games

class Environment:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def State(self):
        return getattr(games, self.game)()

def make(envname, args=None):
    getattr(games, envname)
    return Environment(envname, args)
