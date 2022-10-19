import requests

### Prediction on dict json
response = requests.post(
   "http://127.0.0.1:3000/classify",
   headers={"content-type": "application/json"},
   data="""{"seniority": 10,
"home": "owner",
"time": 36,
"age": 26,
"marital": "married",
"records": "no",
"job": "freelance",
"expenses": 150,
"income": 210.0,
"assets": 10000.0,
"debt": 0.0,
"amount": 5000,
"price": 1400}""",
).text

### Prediction on vector numpy
# response = requests.post(
#    "http://127.0.0.1:3000/classify",
#    headers={"content-type": "application/json"},
#    data = """[
# [3.6e+01, 1.0e+03, 1.0e+04, 0.0e+00, 7.5e+01, 0.0e+00, 0.0e+00,
#        1.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00,
#        1.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 1.0e+00, 0.0e+00,
#        0.0e+00, 0.0e+00, 0.0e+00, 1.4e+03, 1.0e+00, 0.0e+00, 1.0e+01,
#        3.6e+01]
# ]""",
# ).text

print(response)