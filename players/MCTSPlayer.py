import numpy as np

from players.AbstractPlayer import AbstractPlayer
from rl.mcts import MCTS
from rl.forest import TicTacToeBoard
from shared import X, O

class MCTSPlayer(AbstractPlayer):
    def __init__(self, is_X: bool, rollout_iterations=1000):
        super().__init__(is_X)
        self.tree = MCTS()
        self.rollout_iterations = rollout_iterations
        self.player = X if is_X else O

    def play_move(self, board):
        node = TicTacToeBoard(tup=tuple(board.reshape(16)), turn=True, winner=None, terminal=False, player=self.player)
        for _ in range(self.rollout_iterations):
            self.tree.do_rollout(node)
        move = self.tree.choose(node)
        row, col = np.not_equal(board, np.array(move.tup).reshape((4, 4))).nonzero()
        return int(row), int(col)