import requests

### For testing serverless requests
# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

#### When testing GatewayAPI (remind that URL must be safely kept from unauthorized users)
url = "https://<URL_GatewayAPI>/API_deploy_test/predict"


data = {'url':'https://upload.wikimedia.org/wikipedia/commons/9/94/Melanerpes-erythrocephalus-003.jpg'}

result = requests.post(url, json=data).json()
print(result)