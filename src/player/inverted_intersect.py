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

        union = []
        greens = colors[Letter.GREEN]
        yellows = colors[Letter.YELLOW]
        grays = colors[Letter.GRAY]
        print(greens)
        print(yellows)
        print(grays)
        if greens:
            green(greens)
        if yellows:
            yellow(yellows)
        if grays:
            gray(grays)

    def green(self, greens):
        pass
    def yellow(self, yellows):
        pass
    def gray(self, grays):
        pass
    