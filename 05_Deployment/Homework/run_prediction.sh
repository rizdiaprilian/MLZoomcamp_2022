
LOCAL_TAG=`date +"%Y-%m-%d-%H-%M-%S"`
export LOCAL_IMAGE_NAME="svizor/zoomcamp-model:${LOCAL_TAG}"

docker build -t ${LOCAL_IMAGE_NAME} .

docker run -it --rm \
    -p 9696:9696 \
    ${LOCAL_IMAGE_NAME}