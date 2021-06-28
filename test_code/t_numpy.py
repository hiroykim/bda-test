import numpy as np

def test_1():
    print("test_1")
    input = np.random.randn(5,5)
    print(input)
    print(input.shape)

def test_2():
    input = shape([1,2],1)
    np.zeros(input)
    print(input)


def main():
    #test_1()
    test_2()


if __name__ == "__main__":
    main()