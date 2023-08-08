'''
Pit two AbstractPlayers against each other.
'''
import numpy as np

from players.MCTSPlayer import MCTSPlayer
from players.MinimaxPlayer import MinimaxPlayer
from players.HumanPlayer import HumanPlayer
from ForestGame import ForestGame


def main():
    p1 = MinimaxPlayer(is_X=True)
    p2 = MCTSPlayer(is_X=False)
    results = []
    rounds = 1
    for i in range(rounds):
        game = ForestGame(p1, p2)
        result = game.play_game()
        results.append(result)

    print(results)

if __name__ == '__main__':
    main()