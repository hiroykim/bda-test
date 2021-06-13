import time
import datetime
from datetime import timedelta, datetime


print("time.time() ->",time.time())
print("time.localtime() ->",time.localtime())
print("time.localtime(time.time()) ->",time.localtime(time.time()))
st = time.localtime()
print("time.localtime().tm_year-> ", time.localtime().tm_year)

# 날짜 포맷 사용
#https://dojang.io/mod/page/view.php?id=2463
date = time.strftime('%Y%m%d%H%M%S', time.localtime())
print("time.strftime('%Y%m%d%H%M%S', time.localtime()) -> :", date)


# db timestamp 사용
print("timestamp: ", datetime.fromtimestamp(time.time()))


# 시간 이동 사용
print("datetime.today():", datetime.today())
today = datetime.today()
d_day = today - timedelta(days=30)
print("before 30days : ", today - timedelta(days=30))
str_d_day = d_day.strftime("%Y-%m-%d 00:00:00")
print("before 30days : ", str_d_day)


print("datetime(2020,1,2): ",datetime(2020,1,2) )

time = datetime.today()
print("datetime.today() : ", time)
timestr = time.strftime("%Y-%m-%d")
year, month, day = timestr.split("-")
print("year, month, day",year, month, day)

from datetime import date

today = date.today()
print("today:", today)
month_ago = today.replace(day = 1) - timedelta(days = 1)
print("month_ago", month_ago)
print("datetime(year,month,1): ",datetime(month_ago.year, month_ago.month, 1).strftime("%Y-%m-%d 00:00:00") )
