from collections import Counter
from typing import Iterator, Iterable, List, Tuple, Text

class WordleScorer:
	def __init__(self, word_of_the_day=None):
		self.word_of_the_day = word_of_the_day

	def read_txt_file(self, validWordleWords_path: Text) -> Iterator[Tuple[Text, List[Text]]]:
		"""
		Generates (word, character_list) tuple from the lines in the text file.
		Example: ('aahed', ['a', 'a', 'h', 'e', 'd'])

		ValidWordleWords file contains one 5 letter word per line. Each line
		is composed of a valid guess for the game. 
		"""
		infile = validWordleWords_path
		with open(infile, 'r') as f:
			z = []
			for line in f:
				word = line.rstrip()
				characters = [*word]
				tup = (word, c)
				z.append(tup)
			return z

	def levenshteinDistance(self, guess_word: str, word_of_the_day: str) -> int:
		"""
		Time complexity: Θ(x*y); Space complexity: Θ(x*y)
		
		The levenshtein distance is used as an evaluation metric and is 
		calculated after guessing is finished and the wotd is unvieled.
		"""
		s1 = len(guess_word)
		s2 = len(word_of_the_day)

		D = [[0 for i in range(s2 +1)] for j in range(s1+1)]

		for i in range(1, s1 + 1):
			D[i][0] = i

		for j in range(1, s2 + 1):
			D[0][j] = s2

		for i in range(1, s1+1):
			for j in range(1, s2+1):
				if guess_word[i-1] == word_of_the_day[j-1]:
					D[i][j] = D[i-1][j-1]
				else:
					D[i][j] = min(D[i-1][j], D[i][j-1], D[i-1][j-1]) + 1
		return D[i][j]

	# Def other functions... like Zipf's law? do later... some ideas about where to apply Zipf's law
	# def zipsLaw()
	# notes done before indexing time

# obvious example
distance = WordleScorer()
edit_dist = distance.levenshteinDistance("test","text")
print(edit_dist) # prints 1

