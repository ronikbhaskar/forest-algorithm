import numpy as np

from players.AbstractPlayer import AbstractPlayer
from shared import X, O, EMPTY, find_winner, to_pretty_string

class ForestGame:
    def __init__(self, player_1: AbstractPlayer, player_2: AbstractPlayer):
        '''Player 1 always starts'''
        self.board = np.zeros((4, 4), dtype=np.int8)
        self.player_1 = player_1
        self.player_2 = player_2

    def _check_playable(self, row: int, col: int):
        if self.board[row, col] != EMPTY:
            print(row, col)
            raise ValueError('Played an unplayable move!')

    def play_game(self):
        turn = X
        while True:
            if turn == X:
                row, col = self.player_1.play_move(self.board)
                self._check_playable(row, col)
                self.board[row, col] = X
                turn = O
            else:
                row, col = self.player_2.play_move(self.board)
                self._check_playable(row, col)
                self.board[row, col] = O
                turn = X
            print(to_pretty_string(self.board))
            winner = find_winner(self.board)
            if winner is not None:
                return winner
