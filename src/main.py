from game.wordle_game import WordleGame

def main():
    wordle_game = WordleGame('game/valid-wordle-words.txt')
    while True:
        wordle_game.play()
        inp = input('Play another round (y/n)? ')
        if inp.lower() != 'y':
            break
        print()

if __name__ == '__main__':
    main()
