'''
Pit two AbstractPlayers against each other.
'''
from MCTSPlayer import MCTSPlayer
from MinimaxPlayer import MinimaxPlayer

def main():
    rounds = 100
    player_1 = MCTSPlayer()
    player_2 = MinimaxPlayer()
    for i in range(rounds):
        
        player_1.play_move()
        

if __name__ == '__main__':
    main()