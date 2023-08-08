import numpy as np

EMPTY = 0
X = 1
O = 2

TO_CHAR = [' ', 'X', 'O']

def idx_to_row_col(idx: int):
    return idx // 4, idx % 4

def row_col_to_idx(row: int, col: int):
    return row * 4 + col

def winning_combos():
    for start in range(0, 16, 4):  # four in a row
        yield (start, start + 1, start + 2, start + 3)
    for start in range(4):  # four in a column
        yield (start, start + 4, start + 8, start + 12)
    # left -> right diagonals
    yield (0, 5, 10, 15)
    yield (1, 6, 11, 12)
    yield (2, 7, 8, 13)
    yield (3, 4, 9, 14)
    # right -> left diagonals
    yield (3, 6, 9, 12)
    yield (2, 5, 8, 15)
    yield (1, 4, 11, 14)
    yield (0, 7, 10, 13)

def find_winner(board: np.ndarray):
    "Returns winner, None if no winner yet, False if a tie"
    tup = board.reshape(16)
    for i1, i2, i3, i4 in winning_combos():
        v1, v2, v3, v4 = tup[i1], tup[i2], tup[i3], tup[i4]
        if X == v1 == v2 == v3 == v4:
            return X
        if O == v1 == v2 == v3 == v4:
            return O
    
    if all(tup != 0): return False
    return None

def to_pretty_string(board):
    tup = board.reshape(16)
    rows = [
        [TO_CHAR[tup[4 * row + col]] for col in range(4)] for row in range(4)
    ]
    return (
        "\n  0 1 2 3\n"
        + "\n".join(str(i) + " " + " ".join(row) for i, row in enumerate(rows))
        + "\n"
    )