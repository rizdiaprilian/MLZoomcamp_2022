import bentoml
from bentoml.io import JSON, NumpyNdarray 
import numpy as np

## First model
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

## Second model
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("Homework_prediction", runners=[model_runner])

# @svc.api(input=NumpyNdarray(shape=(1, 4), dtype=np.float32, enforce_shape=True), output=JSON())
# def classify(vector):
#     prediction = model_runner.predict.run(vector)

#     return prediction

@svc.api(input=NumpyNdarray(shape=(1, 4), dtype=np.float32, enforce_shape=True), output=JSON())
async def classify(vector):
    prediction = model_runner.predict.run(vector)

    return prediction