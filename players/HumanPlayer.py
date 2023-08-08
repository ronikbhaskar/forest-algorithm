from shared import EMPTY, to_pretty_string

from players.AbstractPlayer import AbstractPlayer

class HumanPlayer(AbstractPlayer):
    def play_move(self, board):
        print(to_pretty_string(board))
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        while board[row, col] != EMPTY:
            print("Invalid move")
            row_col = input("enter row,col: ")
            row, col = map(int, row_col.split(","))

        return row, col
        
        