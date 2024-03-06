from enum import Enum
import random

class Wordle:
    wordle_words = None
    tries = None
    word_of_the_day = None

    def __init__(self, file_name):
        self.wordle_words = self.read_file(file_name)

    '''
    Setter that assigns a random wordle word as word of the day

    @return: None
    '''
    def set_word_of_the_day(self):
        self.word_of_the_day = self.wordle_words[
            random.randrange(len(self.wordle_words))]

    '''
    Getter for word of the day

    @return: word of the day
    '''
    def get_word_of_the_day(self):
        return self.word_of_the_day

    '''
    Reads file of wordle words into list

    @param file_name: name of file to read
    @return: list of words in file
    '''
    def read_file(self, file_name):
        w = []
        f = open(file_name, 'r')
        for word in f:
            w.append(word.strip())
        f.close()
        return w

    '''
    Getter for list of wordle words

    @return: list of wordle words
    '''
    def get_words(self):
        return self.wordle_words

    '''
    Checks whether word that player guessed is valid

    @param guess: guessed word (str)
    @return: True if guess is valid, False otherwise
    '''
    def is_valid(self, guess):
        if guess:
            if len(guess) == 5:
                if guess in self.wordle_words:
                    return True
        return False

    '''
    Gives response to the guessed word like in Wordle

    @param guess: guess (str), assumes self.is_valid checked for guess first
    @return: an array denoting correctness of each letter in guess
    '''
    def guess(self, guess):
        word = [letter for letter in self.word_of_the_day]
        guess = guess.lower()
        response = []
        for i in range(5):
            response.append(Letter.GRAY)
            if guess[i] == word[i]:
                response[-1] = Letter.GREEN
                word[i] = '_'
                continue
            for j in range(5):
                if i != j and guess[i] == word[j]:
                    if i == j:
                        response[-1] = Letter.GREEN
                    else:
                        response[-1] = Letter.YELLOW
                    word[j] = '_'
                    break
        return response

class Letter(Enum):
    GRAY = -1
    YELLOW = 0
    GREEN = 1
