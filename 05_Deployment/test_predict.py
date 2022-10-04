import requests

url = 'http://localhost:9696/predict'

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

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print(f"Sending promo email to {customer['customerid']}")
else:
    print(f"Enjoy our service {customer['customerid']}")