import traceback

def test_err(msg):
    try:
        print("try {0}".format(msg))
        1/0
    except:
        print("func error")
        raise Exception()

if __name__=="__main__":
    try:
        test_err("msg")
    except:
        print("main func : {0}".format(traceback.format_exc()))