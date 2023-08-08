from abc import abstractmethod
from typing import Tuple

import numpy as np

class AbstractPlayer:
    def __init__(self, is_X: bool):
        self.is_X = is_X

    @abstractmethod
    def play_move(self, board: np.ndarray) -> Tuple[int, int]:
        # Gives the board as a 4x4 array, returns the index of the move played as a tuple (row, col)
        raise NotImplementedError('Must implement play_move()')