import requests

# URL for flask app
# url = 'http://localhost:9696/predict'

customer = {
    "customerid": "XXXG-00W0",
    "gender": "male",
    "seniorcitizen": 1,
    "partner": "yes",
    "dependents": "no",
    "tenure": 21,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check", 
    "monthlycharges": 30.55,
    "totalcharges": 43.12
}

customer2 = {
    "customerid": "GN-001",
    "gender": "male",
    "seniorcitizen": 1,
    "partner": "yes",
    "dependents": "no",
    "tenure": 12,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check", 
    "monthlycharges": 19.6,
    "totalcharges": 43.7
}
# Predict with Flask
# response = requests.post(url, json=customer).json()
# print(response)

# if response['churn'] == True:
#     print(f"Sending promo email to {customer['customerid']}")
# else:
#     print(f"Enjoy our service {customer['customerid']}")

# #### Predict with FastAPI
# Adding a new record to customers in dict
response = requests.post("http://127.0.0.1:8000/", json=customer).json()
response = requests.post("http://127.0.0.1:8000/", json=customer2).json()

# Querying customer that we just added
response = requests.get("http://127.0.0.1:8000/customer/XXXG-00W0").json()
print(response)

response = requests.get("http://127.0.0.1:8000/customer/GN-001").json()
print(response)

## Querying customer not exist in customer pydantic dict will return not found
response = requests.get("http://127.0.0.1:8000/customer/XXXG-002").json()
print(response)

# Updating record on a customer
print(requests.put("http://127.0.0.1:8000/update/GN-001?paymentmethod=credit_card").json())

# Predicting churn on a customer
response = requests.post("http://127.0.0.1:8000/predict/", json=customer)
if response.status_code != 200:
    print("An error has occured. [Status code", response.status_code, "]")
    print(response.text)
else:
    print(response.json()) #Only convert to Json when status is OK.