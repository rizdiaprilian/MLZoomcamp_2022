### Initial ver
FROM tensorflow/serving:2.7.0

COPY sequentials/efficient-net-dir /models/eff-net/1

ENV MODEL_NAME="eff-net"