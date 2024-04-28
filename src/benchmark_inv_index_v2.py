from player.inverted_index import InvertedIndex
from player.inverted_intersect import InvertedIntersect
from player.wordle_scorer import WordleGuessScorer

from game.wordle import Wordle, Letter

import time

class BenchmarkInvIndex:

    game = None
    wordle_words = None
    inv_index = None
    inv_intersect = None
    opener = None
    num_tries = None
    scorer = None
    last_result = None


    def __init__(self, opener):
        self.game = Wordle()
        self.wordle_words = self.game.get_wordle_words()
        self.inv_index = InvertedIndex()
        self.inv_intersect = InvertedIntersect()
        self.opener = opener

    def set_opener(self, opener):
        self.opener = opener

    def reset_num_tries(self):
        self.num_tries = 0

    def incr_num_tries(self):
        self.num_tries += 1

    def win(self, res):
        for r in res:
            if r != Letter.GREEN:
                return False
        return True

    def play(self, guess, valid_guesses):
        if self.num_tries >= 6:  # check if the number of tries has reached 6
            print(f"Max tries reached for guess: {guess.upper()}. Moving to next word.")
            return
        #print(f"Playing with guess: {guess.upper()}")  # Print the current guess
        self.scorer = WordleGuessScorer(valid_guesses)
        self.last_result = self.game.guess(guess)  # Store the result of the guess
        #print(f"Result of guess '{guess}': {self.last_result}")  # Print the result of the guess
        self.incr_num_tries()
        if self.win(self.last_result):
            print("Word solved!")  # Print when a word is solved
            return
        index = self.inv_index.index(valid_guesses)
        intersection = self.inv_intersect.get_intersection(guess, self.last_result, index, valid_guesses)
        print(f"Valid guesses left: {len(intersection)}")  # Print the number of valid guesses left

        self.scorer.recompute_scoring(intersection, penalty=6)
        guess = self.scorer.get_best_guess(scoring_method='zipf_idf', lambduh=1)[0]
        self.play(guess, intersection)  # Recurse only if not reached max tries

    def benchmark_words_starting_with(self, letter):
        results = {'letter': letter}
        self.reset_num_tries()
        ctr = 0
        w_s_w = self.inv_index.index(self.wordle_words)[letter + '0'][1:]
        print(f"Benchmarking words starting with '{letter.upper()}'...")
        start = time.time()
        solved = 0
        for word in w_s_w:
            print(f"Setting word of the day: {word.upper()}")
            self.game.set_word_of_the_day(word)
            self.play(self.opener, self.wordle_words)
            if self.win(self.last_result):
                solved += 1
            ctr += self.num_tries
            print(f"Word '{word}' solved in {self.num_tries} tries")
            self.reset_num_tries()
        duration = round(time.time() - start, 4)
        num_words = len(w_s_w)
        avg_tries = round(ctr / num_words, 4)
        avg_time = round(duration / num_words, 4)
        success_rate = round(solved / num_words * 100, 2)
        print(f"Completed benchmarking for words starting with '{letter.upper()}'.")

        results.update({
            'num_words': num_words,
            'solved': solved,
            'success_rate': success_rate,
            'total_time': duration,
            'avg_tries': avg_tries,
            'avg_time': avg_time
        })
        return results

    def benchmark_alphabet(self):
        results = []
        print('Playing with opener \'' + self.opener + '\'')
        for letter in (chr(i) for i in range(97, 123)):
            res = self.benchmark_words_starting_with(letter)
            results.append(res)

        # save results to a file
        with open('benchmark_results.txt', 'w') as file:
            file.write('Playing with opener \'' + self.opener + '\'\n\n')
            for res in results:
                file.write(f"Words starting with '{res['letter']}'\n")
                file.write(f"Number of words:\t\t{res['num_words']}\n")
                file.write(f"Number of words successfully solved:\t{res['solved']}\n")
                file.write(f"Percentage of words successfully solved:\t{res['success_rate']}%\n")
                file.write(f"Total time taken:\t\t{res['total_time']}s\n")
                file.write(f"Average tries per word:\t\t{res['avg_tries']}\n")
                file.write(f"Average time per word:\t\t{res['avg_time']}s\n\n")

def main():
    guess = input("Provide your guess: ")
    b = BenchmarkInvIndex(guess.lower())
    b.benchmark_alphabet()

if __name__ == '__main__':
    main()
