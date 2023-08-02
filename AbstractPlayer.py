from abc import abstractmethod

class AbstractPlayer:
    def __init__(self, first: bool):
        self.first = first
        self.marker = 'X' if first else 'O'

    @abstractmethod
    def play_move(self, board):
        # Gives the board as a 4x4 array, returns the index of the move played as a tuple (row, col)
        raise NotImplementedError('Must implement play_move()')