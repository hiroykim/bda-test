import time
import sys
import traceback

print("test")
try :
    for i in range(1,4):
        time.sleep(1)
    1/0
    sys.stdout.write("success")
except Exception:
    sys.stderr.write("fail")
    raise

sys.exit(0)