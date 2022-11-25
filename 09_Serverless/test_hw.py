import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

### Godzilla image
data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

### Dragon logo image example
data_2 = {'url': 'https://themenscentre.ca/wp-content/uploads/2014/11/Dragon-Logo-e1415228111130.jpg'}

result = requests.post(url, json=data).json()
print(result)