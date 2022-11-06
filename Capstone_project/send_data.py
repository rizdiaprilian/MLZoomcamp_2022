import requests

### Prediction on dict json
response = requests.post(
   "http://127.0.0.1:3000/classify",
   headers={"content-type": "application/json"},
   data="""{"age": 41.0,
"anaemia": "no",
"creatinine_phosphokinase": 213,
"diabetes": "no",
"ejection_fraction": 89,
"high_blood_pressure": "no",
"platelets": 234100.0,
"serum_creatinine": 5.6,
"serum_sodium": 120,
"sex": "male",
"smoking": "no",
"time": 23}""",
).text

print(response)
