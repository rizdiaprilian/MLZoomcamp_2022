import requests

url = 'http://localhost:9696/predict'

data = {'url':'https://upload.wikimedia.org/wikipedia/commons/c/c8/Wood_Duck_%28Aix_sponsa%29.jpg'}

result = requests.post(url, json=data).json()
print(result)