import requests

# Initial localhost setting
# url = 'http://localhost:9696/predict'

# Testing port forwarding sent from service/gateway-serve
# url = 'http://localhost:8080/predict'

# Testing request delivered to service via External IP port
# Acting as Load Balancer on EndPoint
url = 'http://<EXTERNAL_IP>/predict'

data = {'url':'https://upload.wikimedia.org/wikipedia/commons/9/94/Melanerpes-erythrocephalus-003.jpg'}

result = requests.post(url, json=data).json()
print(result)