from player.inverted_index import InvertedIndex
from player.inverted_intersect import InvertedIntersect

from game.wordle import Wordle

class BenchmarkInvIndex:
    game = None
    inv_index = None
    inv_intersect = None

    def __init__(self):
        self.game = Wordle()
        self.inv_index = InvertedIndex()
        self.inv_intersect = InvertedIntersect()

    def benchmark(self):
        '''
        Runs benchmark for wordle with inverted index

        @Return: None (output in terminal)
        '''
        

def main():
    
    g = Wordle()
    inv_index = InvertedIndex()
    inv_intersect = InvertedIntersect()
    
    g.set_word_of_the_day('drank')
    res = g.guess('prank')
    inv_index.index(g.get_wordle_words())
    intersected =inv_intersect.get_intersection('prank',
                                                res,
                                                inv_index.get_index(),
                                                g.get_wordle_words())
    

if __name__ == '__main__':
    main()