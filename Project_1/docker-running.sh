docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/sequentials/efficient-net-dir:/models/eff-net/1" \
    -e MODEL_NAME="eff-net" \
    tensorflow/serving:2.7.0

### Docker build -> create image "zoomcamp-eff-net"
docker build -t zoomcamp-eff-net:eff-net-v1  -f image-model.dockerfile .

docker run -it --rm \
    -p 8500:8500 \
    zoomcamp-eff-net:eff-net-v1 

### Docker build -> create image "gateway-eff-net"
docker build -t gateway-eff-net:eff-net-v1  -f image-gateway.dockerfile .

docker run -it --rm \
    -p 9696:9696 \
    gateway-eff-net:eff-net-v1