# FROM public.ecr.aws/lambda/python:3.9
FROM svizor42/zoomcamp-dino-dragon-lambda:v2

RUN pip install keras-image-helper

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl


COPY lambda_function.py .
ENV MODEL_NAME=dino-vs-dragon-v2.tflite
CMD [ "lambda_function.lambda_handler" ]