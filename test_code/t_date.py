import time
import datetime
from datetime import timedelta, datetime


print(time.time())
print(time.localtime())
print(time.localtime(time.time()))
st = time.localtime()
print(st.tm_year)

# 날짜 포맷 사용
#https://dojang.io/mod/page/view.php?id=2463
date = time.strftime('%Y%m%d%H%M%S', time.localtime())
print("YYYYMMDDhh24mmss :", date)

# 시간 이동 사용
print(datetime(2020,2,2))
print(datetime.today())
today = datetime.today()
d_day = today - timedelta(days=30)
print(d_day)

# db timestamp 사용
print("timestamp: ", datetime.fromtimestamp(time.time()))
