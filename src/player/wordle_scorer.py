from collections import Counter
from typing import Iterator, Iterable, List, Tuple, Text, Dict, Union
from math import log10

NESTED_DICT = Dict[str, Dict[str, Union[str, float]]]
WORD_TUPLE = Iterable[Tuple[Text, List[str]]]
TOTAL_DOCS = 9615720

class WordleGuessScorer:
	'''
	WordleGuessScorer class for returning the best guess based on
	multiple statistics. 

	Code in the wild:
	>>> game = Wordle() 
	>>> inv_index = InvertedIndex()
	>>> inv_intersect = InvertedIntersection()
	>>> game.set_word_of_the_day('plait)
	>>> words_list = game.get_wordle_words()
	>>> scorer = WordleGuessScorer(words_list)
	>>> best_guess = scorer.get_best_guess(scoring_method='zipf')
	>>> res = game.guess(best_guess)
	>>> words_list = inv_intersect.get_intersection(...)
	>>> scorer.recompute_scoring(words_list)
	>>> best_guess = scorer.get_best_guess()
	'''
	wordle_words = None # changes once the recompute scoring is called
	word_of_the_day = None # constant
	words_tuple = None # changes once the recompute scoring is called
	scores = None # changes once the recompute scoring is called
	idf = None # constant
	zipf_idf = None # constant

	def __init__(self, wordle_words_list):
		self.wordle_words = wordle_words_list
		self.generate_word_tuples()

		self.overall_freq = self.get_overall_freq()
		self.position_scores = {}
		self.pos1_freq = self.get_position_freq(0)
		self.pos2_freq = self.get_position_freq(1)
		self.pos3_freq = self.get_position_freq(2)
		self.pos4_freq = self.get_position_freq(3)
		self.pos5_freq = self.get_position_freq(4)

		self.calculate_word_scores()
		self.read_idf_scores()

	def set_word_of_the_day(self, word):
		'''
		Setter for word of the day (for evaluation purposes)
		'''
		self.word_of_the_day = word

	def generate_word_tuples(self) -> WORD_TUPLE:
		"""
		Generates (word, character_list) tuple from the wordle text file.
		
		Ex: ('aahed', ['a', 'a', 'h', 'e', 'd'])
		"""
		self.words_tuple = [(word, [*word]) for word in self.wordle_words]		

	def read_idf_scores(self, path="game/five_letter_words_freq.txt") -> Dict:
		"""
		Generates a dictionary {'word': idf_score} from the text file.
		
		@param path: text file from large corpora, filtered for five letter words
		@return: dictionary with computed idf score
		"""
		doc_freq = {}
		idf = {}
		with open(path, 'r') as infile:
			for line in infile:
				word,freq = line.split()
				doc_freq[word] = freq
		for word in self.scores.keys():
			# add-one smoothing 
			if word not in doc_freq.keys():
				doc_freq[word] = 1
		for word,freq in doc_freq.items():
			total_docs = TOTAL_DOCS 
			idf[word] = log10(total_docs/int(freq))
		self.idf = idf

	def count_words(self) -> int:
		'''
		Method that counts the total amount of words. 
		'''
		return len(self.words_tuple)

	def levenshtein_distance(self, guess_word: str) -> int:
		"""
		The levenshtein distance is used as an evaluation metric after 
		guessing is finished and wotd is revealed.

		@param guess_word: checks for a guess word to compare to the wotd
		@return: integer that tells you how different two strings are
		"""
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
		"""
		Method for calculating the overall letter distribution. 

		@return: dictionary in descending order
		"""
		counts = Counter()
		for word,_ in self.words_tuple:
			counts.update(word)
		sorted_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sorted_freq = sorted_freq
		self.overall_scores = self.letter_scores()
		return sorted_freq

	def get_position_freq(self, position: int) -> Dict:
		"""
		Method for calculating the letter frequency based on its position.

		@return: dictionary in descending order
		"""
		counts = Counter()
		for word,_ in self.words_tuple:
			counts.update(word[position])
		sorted_freq = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
		self.sorted_freq = sorted_freq
		self.position_scores[position] = self.letter_scores()
		return sorted_freq

	def letter_scores(self) -> NESTED_DICT:
		"""
		Generates a score for each letter distribution from the sorted_freq attribute. 

		A low score is desirable and indicates the penalty for choosing the letter. The 
		score is incremented with each for loop by adding the previous zipf score.

		Penalty = 1 / (rank + β) : where β is letter frequency/total frequency

		@return: Letter distribution
		Ex: Fifth position
			{'s': {'relative_freq': 0.29209, 'zipf': 1.29209, 'score': 1.29209},
			 'e': {'relative_freq': 0.11706, 'zipf': 0.61706, 'score': 1.90915},
			...
			 'j': {'relative_freq': 0.0002, 'zipf': 0.03866, 'score': 4.85442}}		
		"""
		total_freq = sum(self.sorted_freq.values())
		scores = {}
		rank = 1
		accumulator = 0
		for letter, freq in self.sorted_freq.items():
			scores[letter] = {}
			scores[letter]['relative_freq'] = round(freq/total_freq, 5)
			scores[letter]['zipf'] = round(1 / rank + scores[letter]['relative_freq'], 5)
			rank += 1
			scores[letter]['score'] = round(accumulator + scores[letter]['zipf'], 5)
			accumulator = scores[letter]['score']
		return scores

	def calculate_word_scores(self, penalty=6.0) -> Dict[str, float]:
		"""
		Calculates the scores for each word based off their letter distribution.

		@param penalty: Hyperparam set at 6 to reward heterograms (no letter occurs more than once). 
						Decrease the penalty to instead reward words that share common letters.
		@return: dictionary of word scores
		"""
		self.penalty = penalty
		scores_tup = []
		for word in self.words_tuple:
			score = 0
			if len(set(word[0])) < 5:
				score += self.penalty
			for idx, char in enumerate(word[0]):
				score += self.overall_scores[char]['score']
				if idx == 0:
					score += self.position_scores[0][char]['score']
				elif idx == 1:
					score += self.position_scores[1][char]['score']
				elif idx == 2:
					score += self.position_scores[2][char]['score']
				elif idx == 3:
					score += self.position_scores[3][char]['score']
				elif idx == 4:
					score += self.position_scores[4][char]['score']
				tup = (word[0], round(score, 5))
			scores_tup.append(tup)
		unsorted_scores = dict(scores_tup)
		scores = dict(sorted(unsorted_scores.items(), key=lambda x: x[1], reverse=False))
		self.scores = scores
		return scores

	def calculate_zipf_idf(self, reg=None) -> Dict[str, float]:
		"""
		Calculates a modified TF-IDF score, based on its penalty and inverse 
		document frequency, which generates a new score that boosts common words. 

		@return: dictionary containing TF-IDF scores for each word
		"""
		if reg != None:
			lamb = reg
		else:
			lamb = 1
		zipf_idf = {}
		for word in self.scores:
			if word in self.idf:
				zipf_idf[word] = self.scores[word] + (lamb * self.idf[word])
		sort_zipf_idf = dict(sorted(zipf_idf.items(), key=lambda x: x[1], reverse=False))
		self.zipf_idf = sort_zipf_idf
		return sort_zipf_idf

	def get_best_guess(self, topk=1, scoring_method="zipf", lambduh=None) -> List[str]:
		"""
		Getter method for returning the best guess based on the specified scoring.

		@param topk: number of top words to return
		@param scoring_method: default value zipf (also zipf_idf)
		@return: list of best guesses
		"""
		if scoring_method == "zipf_idf":
			scores = self.calculate_zipf_idf(lambduh)
		else:
			scores = self.scores
		return list(scores.keys())[:topk]

	def recompute_scoring(self, intersected_words=None, penalty=6) -> None:
		"""
		Method for returning the best guess based on a list of intersected words.

		@param intersected_words: 
		@param penalty: option to increase or decrease penalty for non-heterograms
		@return : None. Call get_best_guess() method after
		"""
		self.wordle_words = intersected_words
		self.generate_word_tuples()
		self.overall_freq = self.get_overall_freq()
		self.pos1_freq = self.get_position_freq(0)
		self.pos2_freq = self.get_position_freq(1)
		self.pos3_freq = self.get_position_freq(2)
		self.pos4_freq = self.get_position_freq(3)
		self.pos5_freq = self.get_position_freq(4)
		self.calculate_word_scores(penalty=penalty)