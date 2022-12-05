import requests
from time import sleep

url = "http://localhost:9696/predict"

client = {"reports": 0, "share": 0.72, "expenditure": 3.45, "owner": "yes"}
# response = requests.post(url, json=client).json()

# print(response)

while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)
