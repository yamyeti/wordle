from collections import Counter
from typing import Iterator, Iterable, List, Tuple, Text, Dict

NestedDict = dict[str, dict[str, float|str, float|str, float]]

class WordleScorer:
	def __init__(self, word_of_the_day=None):
		"""
		WordlerScorer class for returning multiple statistics on 
		the valid-wordle-words file and for evaluating each guess.

		Note: wotd is needed to instantiate the class! ;) 

		Example: 
		>>> scorer = WordleScorer("decay")
		>>> file_path = 'valid-wordle-words.txt'
		>>> words = scorer.readWordleFile(file_path)
		>>> scorer.levenshteinDistance("delay") # prints 1
		>>> freq = scorer.getAllLetterFrequencies(words)
		>>> scorer.letterScore()
		"""
		self.word_of_the_day = word_of_the_day
		if self.word_of_the_day is None:
			raise ValueError("Word of the Day is not provided.")
		self.sort_letters = None

	def readWordleFile(self, validWordleWords_path: Text) -> Iterator[Tuple[Text, List[Text]]]:
		"""
		Generates (word, character_list) tuple from the lines in the text file.
		Example: ('aahed', ['a', 'a', 'h', 'e', 'd'])

		ValidWordleWords file contains one 5 letter word per line. Each line
		is composed of a valid guess for the game. 
		"""
		infile = validWordleWords_path
		with open(infile, 'r') as f:
			words_tup = []
			for line in f:
				word = line.rstrip()
				characters = [*word]
				tup = (word, characters)
				words_tup.append(tup)
			return words_tup

	def countWords(self, words_tup: Iterator[Tuple[Text, List[Text]]]) -> int:
		"""
		Method that counts the total amount of words in the curated list 
		of 5-letter words.
		"""
		count = 0
		for word in words_tup:
			count += 1
		return count

	def levenshteinDistance(self, guess_word: str) -> int:
		"""
		Time complexity: Θ(x*y); Space complexity: Θ(x*y)
		
		The levenshtein distance is used as an evaluation metric and is 
		calculated after guessing is finished and the wotd is revealed.

		>>> print(distance.levenshteinDistance("test","text"))
		>>> 1
		"""
		word_of_the_day = self.word_of_the_day
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

	def getAllLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Counter:
		"""
		Method for calculating the overall letter distribution and returns
		a counter in descending order. 

		All letter frequency methods create the attribute sort_letters to
		populate the letterScore() method to generate a score. 
		"""
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts

	def getFirstLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Counter:
		"""
		Method for calculating the starting letter frequency and returns
		a dictionary in descending order.
		"""
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0][0])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts
	def getSecondLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Dict:
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0][1])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts

	def getThirdLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Counter:
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0][2])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts

	def getFourthLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Counter:
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0][3])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts

	def getLastLetterFrequencies(self, words_tuple: Iterator[Tuple[Text, List[Text]]]) -> Counter:
		letter_counts = Counter()
		for tup in words_tuple:
			letter_counts.update(tup[0][4])
		sort_letters = dict(sorted(letter_counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_letters = sort_letters
		return letter_counts

	def letterScore(self) -> NestedDict:
		"""
		Generates a score for each letter by calling one of the getLetterFrequencies() 
		methods beforehand and using the relative frequencies to scale each score. A 
		low score is desirable and indicates the penalty for choosing the letter. The 
		score is incremented with each for loop by adding the previous zipf score.

		Note: the dictionary must be properly sorted to calculate the scoring! This means 
		using Counter() will incorrectly generate scoring.

		Adjusted Zipf's law: freq ∝ 1 / (rank + β) where β is letter frequency/total frequency

		Ex: {'e': {'relFrq': 0.10037, 'zipf': 0.90879, 'score': 0.90879}, 
			...
			 'q': {'relFrq': 0.00195, 'zipf': 0.03846, 'score': 3.71671}}
		"""
		if self.sort_letters == None:
			raise ValueError("Letter frequencies not calculated. Please call one of the getLetterFrequencies() methods first.")

		total = sum(self.sort_letters.values())
		scores = {}
		rank = 1
		accumulator = 0
		for key, val in self.sort_letters.items():
			scores[key] = {}
			scores[key]['relFrq'] = round(val/total, 5)
			scores[key]['zipf'] = round(1 / (rank + scores[key]['relFrq']), 5)
			rank += 1
			scores[key]['score'] = round(accumulator + scores[key]['zipf'], 5)
			accumulator = scores[key]['score']
		return scores

# make psuedocode for computing the best word wrt the scores
# sum of all scores from zipf's law
	# def AlgorithmForComputingTheBestWord
	# will return a tuple that we can use as a sort of static 
	# dictionary that we can search through to possibly influence 
	# our guessing 
	# def wordScore(self) -> Dict:
	# 	for word in words_tup:
