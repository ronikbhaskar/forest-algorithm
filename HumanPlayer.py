from AbstractPlayer import AbstractPlayer

class HumanPlayer(AbstractPlayer):
    def play_move(self, board):
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 4 * (row - 1) + (col - 1)
        while board[index] is not None:
            print("Invalid move")
            row_col = input("enter row,col: ")
            row, col = map(int, row_col.split(","))
            index = 4 * (row - 1) + (col - 1)

        return row, col
        
        