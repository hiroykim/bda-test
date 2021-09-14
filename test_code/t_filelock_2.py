from time import sleep
from filelock import Timeout, FileLock
import os

lock = FileLock("high_ground.txt.lock", timeout=10)
print("pgm_2--------1----------{}".format(os.getpid()) )
print("lock.is_locked : ", lock.is_locked)

try:
    with lock:
        print("lock.is_locked : ", lock.is_locked)
        print("pgm_2--------2----------{}".format(os.getpid()) )
        open("high_ground.txt", "a").write("You were the chosen pgm_2." + str(os.getpid()) + "\n")
except Timeout:
    print("Timeout occured!! : ", Timeout)
except Exception as e:
    print("Error occured!! : ", e.__class__)
finally:
    print("lock.is_locked : ", lock.is_locked)

print("End Process")


