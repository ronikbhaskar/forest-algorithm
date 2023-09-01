"""
So, I created a game called FOREST. I will explain the game in detail 
later down the line. For now, know that Dawson and I have agreed to
each write a program that plays FOREST as optimally as possible. I'm
creating a minimax solver. Dawson is using Monte Carlo RL. For Dawson,
it's a matter of how much he can train his model. He's guaranteed a model,
and the more time he puts into it, the more it will improve, but it likely
will never be perfect. On the other hand, if I create a model correctly,
then it will play perfectly. If I can write very efficient code, then 
I will have my model. If my implementation is too slow, it will never be
done, and I will have nothing.

I finished mine.

notes:
board is a numpy array of 16 int8's
lookup is a dictionary
we assume X goes first (WLOG)
"""

# imports
import numpy as np
import random
import pickle

# poorly named constants
EMPTY = 0
X = 1
O = 2
BOARD_SIZE = 16

# pre-computed win states
WINNING_PATTERNS = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
        [0, 5, 10, 15],
        [1, 6, 11, 12],
        [2, 7, 8, 13],
        [3, 4, 9, 14],
        [3, 6, 9, 12],
        [2, 5, 8, 15],
        [1, 4, 11, 14],
        [0, 7, 10, 13]
    ],
    dtype=np.int8
)

# pre-computed equivalence classes
EQUIV_CLASSES = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14],
        [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13],
        [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12],
        [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10],
        [14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9],
        [13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8],
        [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
        [11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6],
        [10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5],
        [9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4],
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3],
        [7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2],
        [6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1],
        [5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0],
        [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3],
        [0, 12, 8, 4, 1, 13, 9, 5, 2, 14, 10, 6, 3, 15, 11, 7],
        [4, 0, 12, 8, 5, 1, 13, 9, 6, 2, 14, 10, 7, 3, 15, 11],
        [8, 4, 0, 12, 9, 5, 1, 13, 10, 6, 2, 14, 11, 7, 3, 15],
        [15, 11, 7, 3, 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2],
        [3, 15, 11, 7, 0, 12, 8, 4, 1, 13, 9, 5, 2, 14, 10, 6],
        [7, 3, 15, 11, 4, 0, 12, 8, 5, 1, 13, 9, 6, 2, 14, 10],
        [11, 7, 3, 15, 8, 4, 0, 12, 9, 5, 1, 13, 10, 6, 2, 14],
        [14, 10, 6, 2, 15, 11, 7, 3, 12, 8, 4, 0, 13, 9, 5, 1],
        [2, 14, 10, 6, 3, 15, 11, 7, 0, 12, 8, 4, 1, 13, 9, 5],
        [6, 2, 14, 10, 7, 3, 15, 11, 4, 0, 12, 8, 5, 1, 13, 9],
        [10, 6, 2, 14, 11, 7, 3, 15, 8, 4, 0, 12, 9, 5, 1, 13],
        [13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 12, 8, 4, 0],
        [1, 13, 9, 5, 2, 14, 10, 6, 3, 15, 11, 7, 0, 12, 8, 4],
        [5, 1, 13, 9, 6, 2, 14, 10, 7, 3, 15, 11, 4, 0, 12, 8],
        [9, 5, 1, 13, 10, 6, 2, 14, 11, 7, 3, 15, 8, 4, 0, 12],
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1],
        [13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2],
        [14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3],
        [3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4],
        [0, 3, 2, 1, 12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5],
        [1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6],
        [2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7],
        [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
        [4, 7, 6, 5, 0, 3, 2, 1, 12, 15, 14, 13, 8, 11, 10, 9],
        [5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10],
        [6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12],
        [8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 12, 15, 14, 13],
        [9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14],
        [10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15],
        [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12],
        [15, 3, 7, 11, 14, 2, 6, 10, 13, 1, 5, 9, 12, 0, 4, 8],
        [11, 15, 3, 7, 10, 14, 2, 6, 9, 13, 1, 5, 8, 12, 0, 4],
        [7, 11, 15, 3, 6, 10, 14, 2, 5, 9, 13, 1, 4, 8, 12, 0],
        [0, 4, 8, 12, 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13],
        [12, 0, 4, 8, 15, 3, 7, 11, 14, 2, 6, 10, 13, 1, 5, 9],
        [8, 12, 0, 4, 11, 15, 3, 7, 10, 14, 2, 6, 9, 13, 1, 5],
        [4, 8, 12, 0, 7, 11, 15, 3, 6, 10, 14, 2, 5, 9, 13, 1],
        [1, 5, 9, 13, 0, 4, 8, 12, 3, 7, 11, 15, 2, 6, 10, 14],
        [13, 1, 5, 9, 12, 0, 4, 8, 15, 3, 7, 11, 14, 2, 6, 10],
        [9, 13, 1, 5, 8, 12, 0, 4, 11, 15, 3, 7, 10, 14, 2, 6],
        [5, 9, 13, 1, 4, 8, 12, 0, 7, 11, 15, 3, 6, 10, 14, 2],
        [2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 3, 7, 11, 15],
        [14, 2, 6, 10, 13, 1, 5, 9, 12, 0, 4, 8, 15, 3, 7, 11],
        [10, 14, 2, 6, 9, 13, 1, 5, 8, 12, 0, 4, 11, 15, 3, 7],
        [6, 10, 14, 2, 5, 9, 13, 1, 4, 8, 12, 0, 7, 11, 15, 3],
        [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
        [0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13],
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14],
        [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15],
        [15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8],
        [12, 15, 14, 13, 0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9],
        [13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
        [14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11],
        [11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4],
        [8, 11, 10, 9, 12, 15, 14, 13, 0, 3, 2, 1, 4, 7, 6, 5],
        [9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6],
        [10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7],
        [7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0],
        [4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 0, 3, 2, 1],
        [5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2],
        [6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3],
        [15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0],
        [3, 15, 11, 7, 2, 14, 10, 6, 1, 13, 9, 5, 0, 12, 8, 4],
        [7, 3, 15, 11, 6, 2, 14, 10, 5, 1, 13, 9, 4, 0, 12, 8],
        [11, 7, 3, 15, 10, 6, 2, 14, 9, 5, 1, 13, 8, 4, 0, 12],
        [12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1],
        [0, 12, 8, 4, 3, 15, 11, 7, 2, 14, 10, 6, 1, 13, 9, 5],
        [4, 0, 12, 8, 7, 3, 15, 11, 6, 2, 14, 10, 5, 1, 13, 9],
        [8, 4, 0, 12, 11, 7, 3, 15, 10, 6, 2, 14, 9, 5, 1, 13],
        [13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2],
        [1, 13, 9, 5, 0, 12, 8, 4, 3, 15, 11, 7, 2, 14, 10, 6],
        [5, 1, 13, 9, 4, 0, 12, 8, 7, 3, 15, 11, 6, 2, 14, 10],
        [9, 5, 1, 13, 8, 4, 0, 12, 11, 7, 3, 15, 10, 6, 2, 14],
        [14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3],
        [2, 14, 10, 6, 1, 13, 9, 5, 0, 12, 8, 4, 3, 15, 11, 7],
        [6, 2, 14, 10, 5, 1, 13, 9, 4, 0, 12, 8, 7, 3, 15, 11],
        [10, 6, 2, 14, 9, 5, 1, 13, 8, 4, 0, 12, 11, 7, 3, 15],
        [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
        [15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2],
        [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
        [13, 14, 15, 12, 9, 10, 11, 8, 5, 6, 7, 4, 1, 2, 3, 0],
        [0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7],
        [3, 0, 1, 2, 15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6],
        [2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5],
        [1, 2, 3, 0, 13, 14, 15, 12, 9, 10, 11, 8, 5, 6, 7, 4],
        [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
        [7, 4, 5, 6, 3, 0, 1, 2, 15, 12, 13, 14, 11, 8, 9, 10],
        [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9],
        [5, 6, 7, 4, 1, 2, 3, 0, 13, 14, 15, 12, 9, 10, 11, 8],
        [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15],
        [11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2, 15, 12, 13, 14],
        [10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13],
        [9, 10, 11, 8, 5, 6, 7, 4, 1, 2, 3, 0, 13, 14, 15, 12],
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
        [12, 0, 4, 8, 13, 1, 5, 9, 14, 2, 6, 10, 15, 3, 7, 11],
        [8, 12, 0, 4, 9, 13, 1, 5, 10, 14, 2, 6, 11, 15, 3, 7],
        [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3],
        [3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14],
        [15, 3, 7, 11, 12, 0, 4, 8, 13, 1, 5, 9, 14, 2, 6, 10],
        [11, 15, 3, 7, 8, 12, 0, 4, 9, 13, 1, 5, 10, 14, 2, 6],
        [7, 11, 15, 3, 4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2],
        [2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13],
        [14, 2, 6, 10, 15, 3, 7, 11, 12, 0, 4, 8, 13, 1, 5, 9],
        [10, 14, 2, 6, 11, 15, 3, 7, 8, 12, 0, 4, 9, 13, 1, 5],
        [6, 10, 14, 2, 7, 11, 15, 3, 4, 8, 12, 0, 5, 9, 13, 1],
        [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12],
        [13, 1, 5, 9, 14, 2, 6, 10, 15, 3, 7, 11, 12, 0, 4, 8],
        [9, 13, 1, 5, 10, 14, 2, 6, 11, 15, 3, 7, 8, 12, 0, 4],
        [5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3, 4, 8, 12, 0],
    ],
    dtype=np.int8
)

# utility functions
def blank_board():
    """
    returns: np.array
    get a fresh, blank board
    """
    return np.zeros(16, dtype=np.int8)

def is_win(board, player):
    """
    returns: Bool
    checks if the given player has won
    does not account for potential wins by the other player
    """
    positions = board[WINNING_PATTERNS]
    return np.any(np.all(positions == player, axis=1))

def is_draw(board):
    """
    returns: Bool
    checks that the board is entirely empty
    """
    return np.all(board != EMPTY)

def is_terminal(board):
    """
    returns: Bool
    returns True if someone (anyone) has won or there is a draw
    """
    return is_win(board, X) or is_win(board, O) or is_draw(board)

def show_board(board):
    """
    returns: None
    utility function to print board to stdout
    """
    print(np.reshape(board, (4, 4)))

def make_move(board, move, player):
    """
    returns None
    plays player at the specified move on the given board
    """
    board[move] = player

def get_lookup_key(board, lookup):
    """
    returns: Board | None
    Checks if the given board has an equivalent board in the lookup table
        if so, returns that board
        else, returns None
    """
    equiv_class = board[EQUIV_CLASSES]
    for equiv in equiv_class:
        equiv = equiv.tobytes()
        if equiv in lookup:
            return equiv
    return None

def add_lookup_key(board, lookup, val=[]):
    """
    returns None
    first checks that the board (or an equivalent board) is not already in the lookup table
    if not, it adds the board to the lookup table
    optional: adds the value of val to the lookup table, should be a list of moves from the given board
    """
    if get_lookup_key(board, lookup) is None:
        lookup[board.tobytes()] = val

def key_to_board(key, board):
    """
    converts the lookup key to a board
    copies to (overwrites) a pre-existing board to prevent slow memory allocation
    """
    new_board = np.frombuffer(key, dtype=np.int8)
    np.copyto(board, new_board) # copies in place to prevent more memory allocation

def count_boards_no_optimization(board, cp):
    """
    without any optimizations, how many possible states in the tree?
    hasn't finished running, estimated > 30 trillion
    """

    num_boards = 1

    if is_terminal(board):
        return num_boards
    
    for i in range(BOARD_SIZE):
        if board[i] != EMPTY:
            continue

        board[i] = cp
        num_boards += count_boards_no_optimization(board, X if cp == O else O)
        board[i] = EMPTY
    
    return num_boards

def count_boards(board, lookup, cp):
    """
    just a sanity check to determine how much the equivalence classes and lookup table reduce the search space
    ~77k states
    """
    num_boards = 1

    add_lookup_key(board, lookup)
    if is_terminal(board):
        return num_boards
    
    for i in range(BOARD_SIZE):
        if board[i] != EMPTY:
            continue

        board[i] = cp
        if get_lookup_key(board, lookup) is None:
            num_boards += count_boards(board, lookup, X if cp == O else O)
        board[i] = EMPTY
    
    return num_boards


# important functions

def generate_all_moves(board, lookup, cp):
    """
    fills out the lookup table as a decision tree that hasn't been optimized yet
    """

    moves = []

    add_lookup_key(board, lookup)
    key = get_lookup_key(board, lookup)

    if is_terminal(board):
        lookup[key] = moves
        return 
    
    for i in range(BOARD_SIZE):
        if board[i] != EMPTY:
            continue

        moves.append(i)
        board[i] = cp
        if get_lookup_key(board, lookup) is None:
            generate_all_moves(board, lookup, X if cp == O else O)
        board[i] = EMPTY
    
    lookup[key] = moves

def minimax(board, lookup, best_for_x=True, depth=0):
    """
    implementation of minimax algorithm to train forest model
    """

    key = get_lookup_key(board, lookup) 
    moves = lookup[key]
    key_to_board(key, board)
    
    if len(moves) == 0:
        if is_win(board, X):
            lookup[key] = []
            return 1
        elif is_win(board, O):
            lookup[key] = []
            return -1
        elif is_draw(board):
            lookup[key] = []
            return 0

    best_move = moves[0]
    best_score = -1e6 if best_for_x else 1e6

    for move in moves:
        board[move] = X if best_for_x else O
        score = minimax(board, lookup, best_for_x=not best_for_x, depth=depth+1)
        if best_for_x and score > best_score: # max for X
            best_score = score
            best_move = move
        elif not best_for_x and score < best_score: # min for O
            best_score = score
            best_move = move
        key_to_board(key, board)

    lookup[key] = [best_move]

    return best_score


def main():
    # model plays as X

    with open("lookup_x.pkl", "rb") as f:
        lookup = pickle.load(f)

    board = blank_board()
    print("starting game")

    while 1:
        key = get_lookup_key(board, lookup)
        key_to_board(key, board)
        move = lookup[key][0]
        board[move] = X
        print(np.reshape(board, (4, 4)))

        if is_win(board, X):
            print("X wins")
            break
        elif is_draw(board):
            print("draw")
            break

        move = int(input("enter move: "))
        board[move] = O

        if is_win(board, O):
            print("O wins")
            break
        elif is_draw(board):
            print("draw")
            break

if __name__ == '__main__':
    main()