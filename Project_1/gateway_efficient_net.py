import os
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from proto import np_to_protobuf
from keras_image_helper.base import BasePreprocessor, download_image

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def preprocess_ease(image_urls):
    preprocess_engine = BasePreprocessor((224, 224))
    image_input = download_image(image_urls[0])
    img_main = preprocess_engine.resize_image(image_input)
    X = np.array(img_main)
    X = np.array([X])

    return X


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'eff-net'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_23'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = ['Northern_mockingbird', 'Red_headed_Woodpecker', 'Wood_duck']

def prepare_response(pb_response):
    preds = pb_response.outputs['pred'].float_val
    pred_rounded = [round(num, 5) for num in preds]
    return dict(zip(classes, pred_rounded))


def predict(url):
    X = preprocess_ease(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=30.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    image_urls = [data['url']]
    result = predict(image_urls)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)