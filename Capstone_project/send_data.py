import requests

### Prediction on dict json
response = requests.post(
   "http://127.0.0.1:3000/classify",
   headers={"content-type": "application/json"},
   data="""{"age": 80.0,
"anaemia": "no",
"creatinine_phosphokinase": 148,
"diabetes": "yes",
"ejection_fraction": 38,
"high_blood_pressure": "no",
"platelets": 149000.0,
"serum_creatinine": 1.9,
"serum_sodium": 144,
"sex": "male",
"smoking": "yes",
"time": 23}""",
).text

{'age': 80.0,
 'anaemia': 'no',
 'creatinine_phosphokinase': 148,
 'diabetes': 'yes',
 'ejection_fraction': 38,
 'high_blood_pressure': 'no',
 'platelets': 149000.0,
 'serum_creatinine': 1.9,
 'serum_sodium': 144,
 'sex': 'male',
 'smoking': 'yes',
 'time': 23}

print(response)
