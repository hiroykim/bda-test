import traceback

def test():
    test_err("test")

def test_err(msg):
    try:
        print("try {0}".format(msg))
        1/0
    except:
        print("func error")
        raise Exception("hahahja error")

def test_err2(msg):
    print("try1 {0}".format(msg))
    1/0
    print("try2 {0}".format(msg))


if __name__=="__main__":
    try:
        test()
    except:
        print("main func : {0}".format(traceback.format_exc()))