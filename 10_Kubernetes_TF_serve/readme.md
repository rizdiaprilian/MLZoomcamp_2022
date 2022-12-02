### Kubernetes and TensorFlow Serve

This module covers the practices of ramping up prediction in packages with Kubernetes and TensorFlow Serve.

### Kubernetes

### TensorFlow Serve

For more detail on serving TensorFlow model with docker, please read [here](https://www.tensorflow.org/tfx/serving/docker)

Since my practice took a different approach with what covered in video 10.2, library `keras_image-helper` is not used to preprocess image. 

Also, after many times failing to run docker in bash terminal VSCode, luckily Ubuntu can get this works.

#### Sequential model (using feature extraction from VGG16)

```
docker run -it --rm -p 8500:8500 -v "$(pwd)/seq-model-dir:/models/sequential/1" -e MODEL_NAME="sequential" tensorflow/serving:2.7.0
```

#### EfficientNet model

```
docker run -it --rm -p 8500:8500 -v "$(pwd)/efficient-net-dir:/models/eff-net/1" -e MODEL_NAME="eff-net" tensorflow/serving:2.7.0
```

#### Docker

Testing service with docker

```
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/sequentials/efficient-net-dir:/models/eff-net/1" \
    -e MODEL_NAME="eff-net" \
    tensorflow/serving:2.7.0
```

Building a docker image (note that the directory `efficient-net-dir` is a subdirectory from `sequentials`)

```
docker build -t zoomcamp-eff-net:eff-net-v1  -f image-model.dockerfile
```

Do the same with gateway

```
docker build -t gateway-eff-net:eff-net-v1  -f image-gateway.dockerfile .
```

As there are two images, we want to bring two different application accessible to each other. To achieve this, `docker-compose.yaml` is the most adequate solution.




```