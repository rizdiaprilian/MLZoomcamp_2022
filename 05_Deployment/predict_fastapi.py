from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# model_file = "model_C=2.0.bin"

# class Customer(BaseModel):
#     customer_input: dict

# class Churn_Prob(Customer):
#     churn_output: dict

# with open(model_file, "rb") as f_in:
#     dv, model = pickle.load(f_in)

# def predict_single(customer, dv, model):
#     # turn this customer into a feature matrix
#     X = dv.transform([customer])
#     # probabilty that this customer churns
#     y_pred = model.predict_proba(X)[:,1]
#     return y_pred

app = FastAPI()

@app.get("/ping")
def pong():
    return {"ping": "pong!"}


# @app.post("/predict", response_model=Churn_Prob, status_code=200)
# def get_prediction(input_customer: Customer):
#     customer = input_customer.customer_input
#     prediction = predict_single(customer, dv, model)
#     churn = prediction >= 0.5

#     response_object = {
#         "Customer": customer, 
#         "Churn_prob": float(prediction),
#         "Churn": bool(churn)}

#     return response_object