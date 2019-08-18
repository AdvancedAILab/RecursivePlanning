import modules.GameImplementation.games as games

class TicTacToe:
    @staticmethod
    def State():
        return games.TicTacToe()

class Reversi:
    @staticmethod
    def State():
        return games.Reversi()

def make(envname):
    if envname == 'TicTacToe':
        return TicTacToe()
    if envname == 'Reversi':
        return Reversi()
