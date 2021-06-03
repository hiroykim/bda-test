import time
import datetime
from datetime import timedelta


print(time.time())
print(time.localtime())
print(time.localtime(time.time()))

#https://dojang.io/mod/page/view.php?id=2463
date = time.strftime('%Y%m%d%H%M%S', time.localtime())
print("YYYYMMDDhh24mmss :", date)

print(datetime.datetime(2020,2,2))
print(datetime.datetime.today())
today = datetime.datetime.today()
d_day = today - timedelta(days=30)
print(d_day)
