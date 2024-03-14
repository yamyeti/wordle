from game.wordle_game import WordleGame

def main():
    # new wordle game object
    wordle_game = WordleGame('game/valid-wordle-words.txt')

    # play as many rounds as player likes
    while True:
        wordle_game.play()
        inp = input('Play another round (y/n)? ')
        if inp.lower() != 'y':
            break
        print()

if __name__ == '__main__':
    main()
