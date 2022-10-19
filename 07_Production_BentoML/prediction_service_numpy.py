import bentoml
from bentoml.io import JSON, NumpyNdarray 
from pydantic import BaseModel, ValidationError, validator
import numpy as np


class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


model_ref = bentoml.xgboost.get("credit_scoring_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(-1, 29), dtype=np.float32, enforce_shape=True), output=JSON())
def classify(vector):
    prediction = model_runner.predict.run(vector)

    result = prediction[0]

    if result > 0.5:
        return {"status": "Declined"}
    elif result > 0.25:
        return {"status": "Maybe"}
    else:
        return {"status": "Approved"}
