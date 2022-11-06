import numpy as np
from locust import task
from locust import between
from locust import HttpUser

sample = {"age": 41.0,
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
"time": 23
}

class PatientInput(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000
        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)
    