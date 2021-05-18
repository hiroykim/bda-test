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
    print(*kargs)
    print(a,b,c)
    sample_4(a+1,b+1,c+1, **kargs)


def sample_4(a,b,c, **kargs):
    print("===============================")
    print(kargs)
    print(*kargs)
    print(a,b,c)
    print(kargs.get("key1"))
    print(kargs.get("key2"))
    print(kargs.get("key3"))


def sample_5(*args,**kargs):
    print(kargs)
    print(*kargs)
    #print(**kargs)
    print(args)
    print(args[0]+1,args[1]+1,args[2]+1)
    print(*args)


def sample_6(s=-99, *args,**kargs):
    print(kargs)
    print(*kargs)
    #print(**kargs)
    print(args)
    #print(args[0]+1,args[1]+1)
    print(*args)
    print("s:", s)


if __name__=="__main__":
    options = {"key1": "data1", "key2":"data2", "key3":"data3"}
    sample_1(options)
    sample_2(options)
    #sample_3(options)
    sample_3(11, 22, 33, key1="data1", key2="data2", key3="data3")

    sample_5(11, 22, 33, key1="data1", key2="data2", key3="data3")
    print("--------------------------------->sample6666666666")
    sample_6(key1="data1", key2="data2", key3="data3")

