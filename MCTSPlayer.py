from AbstractPlayer import AbstractPlayer
from rl.mcts import MCTS
from rl.forest import TicTacToeBoard

class MCTSPlayer(AbstractPlayer):
    def __init__(self, first: bool):
        super().__init__(first)
        self.tree = MCTS()
        self.rollout_iterations = 1000

    def play_move(self, board):
        node = TicTacToeBoard(tup=tuple(board), turn=True, winner=None, terminal=False)
        for _ in range(self.rollout_iterations):
            self.tree.do_rollout(node)
        board = self.tree.choose(node)