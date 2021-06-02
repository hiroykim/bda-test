import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from selenium import webdriver

base_url= "https://google.com/search?q="
add_url= input("keyword : ")
url= base_url + quote_plus(add_url)

driver= webdriver.Chrome("../install/chromedriver")

driver.get(url)

html= driver.page_source
