from numpy import add


class X:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return self.add(x, y)

    def a (self, a):
        return a

    def add (self, a, b):
        return a+b
    
    def n (self, a, b, c):
        return a+b+c

def returnout(string):
    return{
        "N": 1,
        "B": 2,
        "C": 3
    }[string]


n = X()

z = n(2,3)

print(z)

print(returnout("C"))