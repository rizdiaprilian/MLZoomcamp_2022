import numpy as np
from locust import task
from locust import between
from locust import HttpUser

sample = {"seniority": 10,
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
"price": 1400
}

class CreditRiskTestUser(HttpUser):
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
    