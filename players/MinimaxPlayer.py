from players.AbstractPlayer import AbstractPlayer
import pickle
from minimax.perfect import get_lookup_key, key_to_board
from shared import idx_to_row_col

class MinimaxPlayer(AbstractPlayer):
    def __init__(self, is_X: bool):
        super().__init__(is_X)
        player = 'x' if is_X else 'o'
        with open(f"minimax/lookup_{player}.pkl", "rb") as f:
            self.lookup = pickle.load(f)

    def play_move(self, board):
        print('MINIMAX')
        print(board)
        key = get_lookup_key(board.reshape(16), self.lookup)
        return idx_to_row_col(self.lookup[key][0])