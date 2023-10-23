class abc:
    def __init__(self,A) -> None:
        self.A = A
        B = [] 
        C = []
        for i in range(5):
            B.append(i)
            C.append(4+i)
        self.B = B
        self.C = C
letter = abc(12)
print(letter.C)