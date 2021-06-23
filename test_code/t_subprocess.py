import subprocess
import time
import traceback

class MyException(Exception):
    """사용자 예외"""
    pass

response_out=""
response_err=""
try:
    s_time = time.localtime()
    sub_pro = subprocess.Popen("python t_subtest.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    response_out, response_err = sub_pro.communicate(timeout=3)
    print("success : {}".format(time.localtime().tm_sec - s_time.tm_sec))
except ZeroDivisionError:
    print("intercept!!")
except ValueError:
    print("timeout : {}".format(time.localtime().tm_sec - s_time.tm_sec))
except:
    raise MyException("MyException")


print("out : {} ".format(response_out))
print("err : {} ".format(response_err))



