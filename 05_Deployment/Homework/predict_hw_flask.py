import pickle
from flask import Flask, request, jsonify


model_file = "model2.bin"
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

app = Flask('hw-program')

@app.route('/predict', methods=['POST']) 
def predict():
    customer = request.get_json()  
    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction), 
        'churn': bool(churn),  
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)