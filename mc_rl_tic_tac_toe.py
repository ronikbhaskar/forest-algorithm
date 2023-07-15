

import numpy as np
import random
import pickle

EMPTY = 0
X = 1
O = 2

EPSILON = 0.1

WINNING_PATTERNS = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [0,3,6],
    [1,4,7],
    [2,5,8],
    [0,4,8],
    [2,4,6]
]

def is_win(board, current_player):

    for win in WINNING_PATTERNS:
        if all((board[index] == current_player for index in win)):
            return True

    return False

def is_draw(board):
    return not np.any(board[:9] == 0)

def is_terminal(board):
    return is_win(board, X) or is_win(board, O) or is_draw(board)

def initial_state():
    return np.zeros(10, dtype=np.int8)

Q_x = dict()
Q_o = dict()

R_x = dict()
R_o = dict()

def fill_dict(board, Q_x, Q_o, cp):
    Q = Q_x if cp == X else Q_o
    
    for i in range(9):
        if board[i] == EMPTY:
            board[9] = i
            Q[board.tobytes()] = 0
            board[i] = cp
            fill_dict(board, Q_x, Q_o, X if cp == O else O)
            board[i] = EMPTY


def fill_moves(board, R, cp):
    for i in range(9):
        if board[i] == EMPTY:
            board[9] = i
            R[board.tobytes()] = []
            board[i] = cp
            fill_moves(board, R, X if cp == O else O)
            board[i] = EMPTY

def pi(board, Q):
    best = 0
    best_val = -1
    all_moves = []
    for i in range(9):
        if board[i] == EMPTY:
            all_moves.append(i)
            board[9] == i
            if Q[board.tobytes()] > best_val:
                best_val = Q[board.tobytes()]
                best = i

    best_moves = []

    for i in range(9):
        if board[i] == EMPTY:
            all_moves.append(i)
            board[9] == i
            if Q[board.tobytes()] == best_val:
                best_moves.append(i)


    # probabilistically return a random
    if random.random() < EPSILON:
        return random.choice(all_moves)
    else:
        return random.choice(best_moves)


def simulate_episode():
    board = initial_state()
    episode_x = []
    episode_o = []
    while not is_terminal(board):
        next_move_x = pi(board, Q_x)
        board[9] = next_move_x
        board[next_move_x] = X
        episode_x.append(board)

        if is_terminal(board):
            return episode_x, episode_o

        next_move_o = pi(board, Q_o)
        board[9] = next_move_o
        board[next_move_x] = O
        episode_o.append(board)

    return episode_x, episode_o

def draw_board(board):
    board = board[:9]
    drawn_board = [
        [" ", " ", "|", " ", " ", "|", " ", " "],
        ["-", "-", "+", "-", "-", "+", "-", "-"],
        [" ", " ", "|", " ", " ", "|", " ", " "],
        ["-", "-", "+", "-", "-", "+", "-", "-"],
        [" ", " ", "|", " ", " ", "|", " ", " "],
    ]

    key = {
        0: " ",
        1: "x",
        2: "o"
    }

    for index, cell in enumerate(board):
        drawn_board[(index // 3) * 2][(index % 3) * 3] = key[int(cell)]
        # x_pos = (index % 3) * 3
        # y_pos = (index // 3) * 2

    for row in drawn_board:
        for char in row:
            print(char, end="")
        print("")
            
# fill_dict(initial_state(), Q_x, Q_o, X)

# fill_moves(initial_state(), R_x, X)
# fill_moves(initial_state(), R_o, X)

with open("qx.pkl", "rb") as f:
    Q_x = pickle.load(f)

with open("qo.pkl", "rb") as f:
    Q_o = pickle.load(f)

with open("rx.pkl", "rb") as f:
    R_x = pickle.load(f)

with open("ro.pkl", "rb") as f:
    R_o = pickle.load(f)


print("starting episode")
for i in range(1):
    episode_x, episode_o = simulate_episode()
    print(episode_x)
    print(episode_o)

