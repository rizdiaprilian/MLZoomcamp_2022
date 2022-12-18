import requests

url = 'http://localhost:9696/predict'

data = {'url':'https://upload.wikimedia.org/wikipedia/commons/9/94/Melanerpes-erythrocephalus-003.jpg'}

result = requests.post(url, json=data).json()
print(result)