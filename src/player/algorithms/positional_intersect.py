class PositionalIntersect:
    args = []
    def __init__(self):

    def append_args()
    def positional_intersect(self, word1, word2, k):
        p1 = self.pos_index[word1]
        p2 = self.pos_index[word2]
        i = 0
        j = 0
        ans = []
            while (i < len(p1) and j < len(p2)):
                if p1[i][0] == p2[j][0]:
                    l = []
                    pp1 = p1[i][1]
                    pp2 = p2[j][1]
                    x = 0
                    y = 0
                    while x < len(pp1):
                        while y < len(pp2):
                            if abs(pp1[x] - pp2[y]) <= k:
                                l.append(pp2[y])
                            elif pp2[y] > pp1[x]:
                                break
                            y += 1
                        while l and abs(l[0] - pp1[x]) > k:
                            l.remove(0)
                        for ps in l:
                            ans.append(Document(self.indices[p1[i][0]], pp1[x], ps))
                            x += 1
                            i += 1
                            j += 1
                elif p1[i][0] < p2[j][0]:
                    i += 1
                else:
                    j += 1
            return ans