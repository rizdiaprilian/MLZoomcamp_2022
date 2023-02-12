from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from typing import Dict, Optional
import uvicorn

# With python 3.8.13

model_file = "model_C=2.0.bin"

app = FastAPI(title="Customer Churn FastAPI")

class Customer(BaseModel):
    # customer_input: dict
    customerid: str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

items = {

}

class Churn_Prob(BaseModel):
    probs: float
    churn_output: bool

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

def predict_single(customer, dv, model):
    # turn this customer into a feature matrix
    X = dv.transform([customer])
    # probabilty that this customer churns
    y_pred = model.predict_proba(X)[:,1]
    return y_pred

def transform_matrix(customer, dv):
    # turn this customer into a feature matrix
    X = dv.transform([customer])
    return X

@app.post("/")
def add_item(customer: Customer) -> Dict[str, Customer]:

    if customer.customerid in items:
        raise HTTPException(status_code=404, detail=f"Customer with {customer.customerid=} already exists.")

    items[customer.customerid] = customer
    return {"added to customers": customer}

@app.get("/customer/{customerid}")
def query_item_by_id(customerid: str) -> Dict[str, Customer]:

    if customerid not in items:
        raise HTTPException(status_code=404, detail=f"Customer with {customerid=} does not exist.")

    return items[customerid]

@app.put("/update/{customerid}")
def update(
        customerid: str,
        contract: Optional[str] = None,
        paperlessbilling: Optional[str] = None,
        paymentmethod: Optional[str] = None,
        monthlycharges: Optional[float] = None,
    ) -> Dict[str, Customer]:

    if customerid not in items:
        raise HTTPException(status_code=404, detail=f"Item with {customerid=} does not exist.")
    if all(info is None for info in (contract, paperlessbilling, paymentmethod, monthlycharges)):
        raise HTTPException(
            status_code=400, detail="No parameters provided for update."
        )

    item = items[customerid]
    if contract is not None:
        item.contract = contract
    if paperlessbilling is not None:
        item.paperlessbilling = paperlessbilling
    if paymentmethod is not None:
        item.paymentmethod = paymentmethod
    if monthlycharges is not None:
        item.monthlycharges = monthlycharges

    return {"updated": item}

@app.delete("/delete/{customerid}")
def delete_item(customerid: str) -> Dict[str, Customer]:

    if customerid not in items:
        raise HTTPException(
            status_code=404, detail=f"Item with {customerid=} does not exist."
        )

    item = items.pop(customerid)
    return {"deleted": item}

@app.post("/predict/", response_model=Churn_Prob, status_code=200)
def transform_predict(input_customer: Customer):
    customer = dict(input_customer) 
    pred = predict_single(customer, dv, model) 
    churn = pred >= 0.5

    return Churn_Prob(probs=float(pred), churn_output=bool(churn))
