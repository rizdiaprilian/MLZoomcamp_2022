import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

# class_names = ['dino', 'dragon']

MODEL_NAME = os.getenv('MODEL_NAME', 'dino-vs-dragon-v2.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']



def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255
    return x


def interpret_image(image_file):
    img = download_image(image_file)
    img = prepare_image(img, target_size=(150, 150))
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred_result =  interpreter.get_tensor(output_index)
    # float_pred = pred_result[0].tolist()
    # return dict(zip(class_names, float_pred))
    return float(pred_result[0,0])

def lambda_handler(event, context):
    url = event['url']
    result = interpret_image(url)
    result_dict = {
        'prediction': result
    }

    return result_dict