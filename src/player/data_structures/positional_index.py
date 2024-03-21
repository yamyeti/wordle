class PositionalIndex:
    indices = None
    pos_index = None

    def __init__(self):
        self.set_indices()

    '''
    Builds inverted index given list of docs

    @Param docs: list of valid wordle words
    @Return: None
    '''
    def build_pos_index(self, docs):
        self.pos_index = {}
        for index in self.indices:
            self.pos_index[index] = []
            for doc in docs:
                if index in doc:
                    t = (doc, [i for i in range(len(doc)) if doc[i] == index])
                    self.pos_index[index].append(t)

    def get_pos_index(self):
        return self.pos_index
    
    def set_indices(self):
        self.indices = []
        for i in range(26):
            self.indices.append(chr(i + 97))

    def print_indices(self):
        print(self.indices)

    def print_pos_index(self):
        for k,v in self.pos_index.items():
            print(k)
            print(v)
