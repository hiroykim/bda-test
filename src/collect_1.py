from bs4 import BeautifulSoup
from selenium import webdriver
import requests

url = "http://www.naver.com"

param ={ "aa": 'bb'}
response = requests.get(url, params=param)

html = response.text
soup = BeautifulSoup(html, 'html.parser')

text = soup.find_all("text")
