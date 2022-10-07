import pickle

model_file = "model1.bin"
dv_file = "dv.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

with open(dv_file, "rb") as f_in:
    dv = pickle.load(f_in)

def predict_single(customer, dv, model):
    # turn this customer into a feature matrix
    X = dv.transform([customer])
    # probabilty that this customer churns
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


def main():
    client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
    pred_prob = predict_single(client, dv, model)
    print(pred_prob)

if __name__ == '__main__':
    main()