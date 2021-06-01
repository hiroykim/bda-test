from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from selenium import webdriver
import pandas as pd

base_url= 'https://www.google.com/search?q='
plus_url=input('keyword : ')
url= base_url + quote_plus(plus_url)

driver= webdriver.Chrome("../install/chromedriver")

driver.get(url)

html = driver.page_source
soup = BeautifulSoup(html)

v= soup.select('.yuRUbf')

print(v)