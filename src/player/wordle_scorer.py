from collections import Counter
from typing import Iterator, Iterable, List, Tuple, Text, Dict, Union

NESTED_DICT = Dict[str, Dict[str, Union[str, float]]]
WORD_TUPLE = Iterable[Tuple[Text, List[str]]]

class WordleScorer:
	'''
	WordleScorer class for returning multiple statistics
	and evaluates each guess.

	Ex:
	>>> game = Wordle() 
	>>> inv_index = InvertedIndex()
	>>> inv_intersect = InvertedIntersection()
	>>> game.set_word_of_the_day('plait)
	>>> scorer = WordleScorer(g.get_wordle_words())
	>>> best_guess = scorer.get_best_guess()
	>>> res = game.guess(best_guess)
	>>> words_list = inv_intersect.get_intersection(...)
	>>> scorer.recompute_scoring(words_list)
	>>> best_guess = scorer.get_best_guess()
	'''
	wordle_words = None
	word_of_the_day = None
	words_tuple = None
	scores = None

	def __init__(self, wordle_words_list):
		self.wordle_words = wordle_words_list
		self.read_words()

		self.overall_freq = self.get_overall_freq()
		self.pos1_freq = self.get_first_freq()
		self.pos2_freq = self.get_second_freq()
		self.pos3_freq = self.get_third_freq()
		self.pos4_freq = self.get_fourth_freq()
		self.pos5_freq = self.get_fifth_freq()
		self.word_scores()

	def set_word_of_the_day(self, word):
		'''
		Setter for word of the day (for evaluation purposes)
		'''
		self.word_of_the_day = word

	def read_words(self) -> WORD_TUPLE:
		'''
		Generates (word, character_list) tuple from the lines in the text file.
		Example: ('aahed', ['a', 'a', 'h', 'e', 'd'])

		'''
		words_tup = []
		for elem in self.wordle_words:
			word = elem
			characters = [*word]
			tup = (word, characters)
			words_tup.append(tup)
		self.words_tuple = words_tup

	def count_words(self) -> int:
		'''
		Method that counts the total amount of words. Intended to
		be called upon after each guess.
		'''
		count = 0
		for word in self.words_tuple:
			count += 1
		return count

	def levenshtein_distance(self, guess_word: str) -> int:
		'''
		The levenshtein distance is used as an evaluation metric after 
		guessing is finished and wotd is revealed.

		@param guess_word: checks for a guess word to compare to the wotd
		@return: integer that tells you how different two strings are
		'''
		s1 = len(guess_word)
		s2 = len(self.word_of_the_day)

		D = [[0 for i in range(s2 +1)] for j in range(s1+1)]

		for i in range(1, s1 + 1):
			D[i][0] = i

		for j in range(1, s2 + 1):
			D[0][j] = s2

		for i in range(1, s1+1):
			for j in range(1, s2+1):
				if guess_word[i-1] == self.word_of_the_day[j-1]:
					D[i][j] = D[i-1][j-1]
				else:
					D[i][j] = min(D[i-1][j], D[i][j-1], D[i-1][j-1]) + 1
		return D[i][j]

	def get_overall_freq(self) -> Dict:
		'''
		Method for calculating the overall letter distribution. 

		@return: dictionary in descending order
		'''
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.overall_scores = self.letter_scores()
		return sort_freq

	def get_first_freq(self) -> Dict:
		'''
		Method for calculating the starting letter frequency. 

		@return: dictionary in descending order
		'''
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0][0])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.pos1_scores= self.letter_scores()
		return sort_freq

	def get_second_freq(self) -> Dict:
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0][1])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.pos2_scores = self.letter_scores()
		return sort_freq

	def get_third_freq(self) -> Dict:
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0][2])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.pos3_scores = self.letter_scores()
		return sort_freq

	def get_fourth_freq(self) -> Dict:
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0][3])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.pos4_scores = self.letter_scores()
		return sort_freq

	def get_fifth_freq(self) -> Dict:
		counts = Counter()
		for tup in self.words_tuple:
			counts.update(tup[0][4])
		sort_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sort_freq = sort_freq
		self.pos5_scores = self.letter_scores()
		return sort_freq

	def letter_scores(self) -> NESTED_DICT:
		'''
		Generates a score for each letter distribution from the sort_freq attribute in 
		each get_frequency method by using the relative frequencies to scale each score.

		A low score is desirable and indicates the penalty for choosing the letter. The 
		score is incremented with each for loop by adding the previous zipf score.

		Adjusted Zipf's law: freq ∝ 1 / (rank + β) where β is letter frequency/total frequency

		Ex: {'e': {'relFrq': 0.10037, 'zipf': 0.90879, 'score': 0.90879}, 
			...
			 'q': {'relFrq': 0.00195, 'zipf': 0.03846, 'score': 3.71671}}
		'''
		total = sum(self.sort_freq.values())
		scores = {}
		rank = 1
		accumulator = 0
		for key, val in self.sort_freq.items():
			scores[key] = {}
			scores[key]['relFrq'] = round(val/total, 5)
			scores[key]['zipf'] = round(1 / (rank + scores[key]['relFrq']), 5)
			rank += 1
			scores[key]['score'] = round(accumulator + scores[key]['zipf'], 5)
			accumulator = scores[key]['score']
		return scores

	def word_scores(self, penalty=3.0) -> Dict:
		self.penalty = penalty
		scores_tup = []
		for word in self.words_tuple:
			score = 0
			if len(set(word[0])) < 5:
				score += self.penalty
			for idx, char in enumerate(word[0]):
				score += self.overall_scores[char]['score']
				if idx == 0:
					score += self.pos1_scores[char]['score']
				elif idx == 1:
					score += self.pos2_scores[char]['score']
				elif idx == 2:
					score += self.pos3_scores[char]['score']
				elif idx == 3:
					score += self.pos4_scores[char]['score']
				elif idx == 4:
					score += self.pos5_scores[char]['score']
				tup = (word[0], round(score, 5))
			scores_tup.append(tup)
		unsorted_scores = dict(scores_tup)
		scores = dict(sorted(unsorted_scores.items(), key=lambda x: x[1], reverse=False))
		self.scores = scores
		return scores

	def get_best_guess(self, topk=1) ->List:
		'''
		Getter method for the best guess

		@param topk: optional for topk best words  
		@return: best guess
		'''
		return " ".join(list(self.scores)[:topk])

	def recompute_scoring(self, intersected_words,penalty=5) -> None:
		self.wordle_words = intersected_words
		self.read_words()
		self.overall_freq = self.get_overall_freq()
		self.pos1_freq = self.get_first_freq()
		self.pos2_freq = self.get_second_freq()
		self.pos3_freq = self.get_third_freq()
		self.pos4_freq = self.get_fourth_freq()
		self.pos5_freq = self.get_fifth_freq()
		self.word_scores(penalty=penalty)