
class bndresult:
    def __init__(self, x1, y1, x2, y2, categories=None, confidence=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.categories = categories
        if confidence is not None:
            self.confidence = round(confidence, 2)
        else:
            self.confidence = None

    def __str__(self):
        return f"{self.categories} {self.confidence} {self.x1} {self.y1} {self.x2} {self.y2}"
    def __eq__(self, other):
        return self.x1 == other.x1 and self.x2 == other.x2 and self.y1 == other.y1 and self.y2 == other.y2

if __name__ == '__main__':
    t = [bndresult(1, 2, 3, 4, '5', 1), bndresult(1, 2, 3, 4, '5', 2)]
    c = bndresult(1, 2, 3, 4, '5', 1)
    # print(str(t[0]))
    # print(t[1])
    # t = sorted(t, key=lambda bndresult: bndresult.confidence, reverse=True)
    # print(t[0])
    # print(t[1])
    print(c in t)
