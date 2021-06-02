import requests
from bs4 import BeautifulSoup

url = 'https://section.blog.naver.com/Search/Post.nhn?pageNo=1&rangeType=ALL&orderBy=sim&keyword=%ED%8C%8C%EC%9D%B4%EC%8D%AC'

params = {
    'pageNo' : 1,
    'rangeType' : 'ALL',
    'orderBy' : 'sim',
    'keyword' : '파이썬'
}

response = requests.get('https://section.blog.naver.com/Search/Post.nhn', params=params)

html = response.text
print(html)
soup= BeautifulSoup(html, 'html.parser')
print("============================================")
print(soup.find_all("script"))
'''
for item in soup.head.children:
    print(item)
'''