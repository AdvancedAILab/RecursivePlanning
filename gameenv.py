import modules.GameImplementation.tictactoe as tictactoe
import modules.GameImplementation.reversi as reversi

class TicTacToe:
    @staticmethod
    def State():
        return tictactoe.TicTacToe()

class Reversi:
    @staticmethod
    def State():
        return reversi.Reversi()

def make(envname):
    if envname == 'TicTacToe':
        return TicTacToe()
    if envname == 'Reversi':
        return Reversi()