

def sample_1(a=1, b=10, c=100, d=1000):
    print("a : {}".format(a))
    print("b : {}".format(b))
    print("c : {}".format(c))
    print("d : {}".format(d))


if __name__=="__main__":
    sample_1()
    sample_1(a=1, b=2)
    sample_1(1, 2, 10, 100)
    sample_1(a=1, b=2, c=10, d=4000)