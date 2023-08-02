from AbstractPlayer import AbstractPlayer
import pickle
from minimax.perfect import get_lookup_key, key_to_board

class MinimaxPlayer(AbstractPlayer):
    def __init__(self, first: bool):
        super().__init__(first)
        player = 'x' if first else 'o'
        with open(f"minimax/lookup_{player}.pkl", "rb") as f:
            self.lookup = pickle.load(f)

    def play_move(self, board):
        key = get_lookup_key(board, board.reshape(16))
        key_to_board(key, board)
        return self.lookup[key][0]