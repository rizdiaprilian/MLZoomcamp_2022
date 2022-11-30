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