from email.mime import application
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, ValidationError, validator

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

    # @validator('expenses')
    # def expenses_alphanumeric(cls, v):
    #     assert v.isalnum(), 'must be alphanumeric'
    #     return v

    # @validator('income')
    # def income_alphanumeric(cls, v):
    #     assert v.isalnum(), 'must be alphanumeric'
    #     return v

model_ref = bentoml.xgboost.get("credit_scoring_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

# @svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
# def classify(credit_application):
#     application_data = credit_application.dict()
#     vector = dv.transform(application_data)
#     prediction = model_runner.predict.run(vector)

#     result = prediction[0]

#     if result > 0.5:
#         return {"status": "Declined"}
#     elif result > 0.25:
#         return {"status": "Maybe"}
#     else:
#         return {"status": "Approved"}

@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
async def classify(credit_application):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)

    result = prediction[0]

    if result > 0.5:
        return {"status": "Declined"}
    elif result > 0.25:
        return {"status": "Maybe"}
    else:
        return {"status": "Approved"}
