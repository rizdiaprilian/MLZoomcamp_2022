import requests

url = 'http://localhost:9696/predict'

# url = 'http://localhost:8080/predict'

# Feel free to change url. Lists of urls housing three specieses are available in `list_urls_bird.txt`
data = {'url':'https://upload.wikimedia.org/wikipedia/commons/c/c8/Wood_Duck_%28Aix_sponsa%29.jpg'}

result = requests.post(url, json=data).json()
print(result)