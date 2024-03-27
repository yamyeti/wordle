from game.wordle import Letter

class InvertedIntersect:

    def __init__(self):
        pass
        
    def intersect(self, guess, response, inv_index):
        '''
        Returns intersection based on guess (the color of letters)

        @Param guess: the five letter string guess
        @Param response: color encodings of letters
        @Param pos_index: inverted index
        @Return: the intersection as a list
        '''
        colors = {}
        for color in Letter:
            colors[color] = []
        for i in range(5):
            index = guess[i] + str(i)
            colors[response[i]].append((index, inv_index[index][0]))
        print(colors)
        for i in range(colors[Letter.GREEN]):

        union = []
        greens = colors[Letter.GREEN]
        if greens:
            green(greens)
                


    def green(self, green):
        if green(greens)
        pass
