import numpy as np
import os
from io import BytesIO
from urllib import request
from PIL import Image
import tflite_runtime.interpreter as tflite

class_names = ['Northern_mockingbird', 'Red_headed_Woodpecker', 'Wood_duck']

MODEL_NAME = os.getenv('MODEL_NAME', 'bird-efficient-net.tflite')
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

def interpret_image(image_file):
    ### Load an Image Red_headed_woodpecker
    img = download_image(image_file)
    img = prepare_image(img, target_size=(260, 260))
    ### Remind that rescaling input is skipped for EfficientNet
    x = np.array(img, dtype='float32')
    X = np.array([x])
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred_result =  interpreter.get_tensor(output_index)
    pred_result = pred_result[0].tolist()
    pred_result = [round(num, 5) for num in pred_result]

    return dict(zip(class_names, pred_result))

def lambda_handler(event, context):
    url = event['url']
    result = interpret_image(url)
    return result