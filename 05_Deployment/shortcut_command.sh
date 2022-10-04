
# Enter into bash mode
docker run -it --rm --entrypoint=bash churn-prediction
gunicorn --bind=0.0.0.0:9696 predict_:app

# Alternative, run directly from image
docker run -it -p 9696:9696 churn-prediction:latest