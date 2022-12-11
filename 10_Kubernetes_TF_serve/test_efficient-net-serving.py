import requests

url = 'http://localhost:9696/predict'

data = {'url':'https://upload.wikimedia.org/wikipedia/commons/c/cf/Northern_Mocking_bird_-_Mimus_polyglottos.JPG'}

result = requests.post(url, json=data).json()
print(result)