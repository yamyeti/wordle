from game.wordle import *

class WordleGame:
    game = None
    player = None
    tries = None

    def __init__(self):
        self.game = Wordle()

    '''
    Getter for Wordle object

    @return: Wordle object
    '''
    def get_game(self):
        return self.game

    '''
    Emulates a game of Wordle, i.e. guess word of the day

    @return: None
    '''
    def play(self):
        self.tries = 6

        print('-' * 70)
        print('Wordle: Guess the Word of the Day!')
        print('-' * 70)
        inp = input('Manually set word of the day (y/n)? ')
        if inp == 'y':
            word = None
            while not self.game.is_valid_word(word):
                word = input('Enter a word of the day (invalid words are ignored): ')
            self.game.set_word_of_the_day(word)
        else:
            self.game.random_word_of_the_day()
        print('-' * 70)
        
        while self.tries != 0:
            print('Number of tries remaining: ' + str(self.tries))
            print('Word of the day (for debugging): ' +
                  self.game.get_word_of_the_day())
            guess = None
            while not self.game.is_valid_word(guess):
                guess = input("Guess (invalid words are ignored): ")
            self.game.guess(guess)
            response = self.game.guess(guess)
            print(str([res.name for res in response]) + '\n')
            if self.player_win(response):
                print('Correct!')
                return
            self.tries -= 1

        print('You have 0 tries remaining.')
        print('The word of the day was \"' +
              self.game.get_word_of_the_day() + '\".')

    '''
    Checks if player guessed the word

    @return: True if player guessed the word, False otherwise
    '''
    def player_win(self, response):
        for color in response:
            if color != Letter.GREEN:
                return False
        return True
