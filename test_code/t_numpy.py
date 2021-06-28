import numpy as np

def test_1():
    print("==================test_1======================")
    input = np.random.randn(5,5)
    print(input)
    print(input.shape)

def test_2():
    print("==================test_2======================")
    input = np.array([[1,2,3],[4,5,6]])
    input = input.reshape(2,-1)
    print(input)
    print(input.shape)

def test_3():
    print("==================test_3======================")
    t_array = np.linspace(0,100,49).astype(int)
    print(t_array)

def test_4():
    print("==================test_4======================")
    t_array = np.arange(0, 100, 2)
    print(t_array)

def test_5():
    print("==================test_5======================")
    n_list = [[[1,2],[2,3],[4,6],[8,9]]]
    input = np.array(n_list)
    print(input)
    print(input.shape)
    input = input.flatten()
    print(np.sort(input)[::-1])
    print(input)
    print(input.shape)

def test_6():
    print("==================test_6======================")
    n_list = [[[1, 2], [2, 3], [4, 6], [8, 9]]]
    input = np.array(n_list)
    print(input)
    print(input.sum())

def main():
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()


if __name__ == "__main__":
    main()