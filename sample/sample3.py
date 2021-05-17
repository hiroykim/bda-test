def sample_1(kargs):
    print(type(kargs))
    print(kargs)

def sample_2(*kargs):
    print(type(kargs))
    print(type(kargs[0]))
    print(kargs)
    print(kargs[0])

def sample_3(a,b,c,**kargs):
    print(kargs)
    print(a,b,c)



if __name__=="__main__":
    options = {"key1": "data1", "key2":"data2", "key3":"data3"}
    sample_1(options)
    sample_2(options)
    #sample_3(options)
    sample_3(11,22,33,key1="data1", key2="data2", key3="data3")
