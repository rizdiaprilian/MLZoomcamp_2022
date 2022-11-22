import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow.lite as tflite
import os


class_names = ['Northern_mockingbird', 'Red_headed_Woodpecker', 'Wood_duck']

interpreter = tflite.Interpreter(model_path='duck-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def interpret_image(image_file):
    ### Load an Image Red_headed_woodpecker
    img = load_img(image_file, target_size=(260, 260))

    ### Remind that rescaling input is skipped for EfficientNet
    x = np.array(img, dtype='float32')
    X = np.array([x])
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred_result =  interpreter.get_tensor(output_index)
    
    return dict(zip(class_names, pred_result[0]))

def lambda_handler(event, context):
    url = event['url']
    result = interpret_image(url)
    return result