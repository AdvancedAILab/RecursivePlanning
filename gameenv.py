import modules.GameImplementation.tictactoe as tictactoe
#import modules.GameImplementation.reversi as reversi

class TicTacToe:
    @staticmethod
    def State():
        return tictactoe.State()

class Reversi:
    @staticmethod
    def State():
        return reversi.State()

def make(envname):
    if envname == 'TicTacToe':
        return TicTacToe()
    if envname == 'Reversi':
        return Reversi()