from time import sleep
from filelock import Timeout, FileLock
import os

lock = FileLock("high_ground.txt.lock", timeout=2)
print("pgm_1--------1----------{}".format(os.getpid()) )
sleep(1)

print("lock.is_locked : ", lock.is_locked)
with lock:
    print("lock.is_locked : ", lock.is_locked)
    sleep(2)
    print("pgm_1--------2----------{}".format(os.getpid()) )
    open("high_ground.txt", "a").write("You were the chosen pgm_1." + str(os.getpid()) + "\n")
    print("lock.is_locked_1 : ", lock.is_locked)
    print("lock.acquire() : ", lock.acquire())
    print("lock.is_locked_2 : ", lock.is_locked)


print("lock.acquire() : ", lock.acquire())
print("lock.is_locked_3 : ", lock.is_locked)
lock.release()
lock.release()
print("lock.is_locked_4 : ", lock.is_locked)
