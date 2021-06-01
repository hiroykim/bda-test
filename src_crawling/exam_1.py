import requests

response = requests.get('https://www.naver.com/')


print(response.status_code)
print(response.text)
if 0:
    print(response.json())
    print(response.content)
    print(response.status_code)
