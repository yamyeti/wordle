from player.data_structures.positional_index import PositionalIndex

from game.wordle import Wordle

def main():
    g = Wordle()
    pi = PositionalIndex()

    pi.build_pos_index(g.get_wordle_words())
    pi.print_pos_index()

if __name__ == '__main__':
    main()
