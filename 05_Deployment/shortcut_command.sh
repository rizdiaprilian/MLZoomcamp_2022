### Running Flask app with Docker
# Enter into bash mode
docker run -it --rm --entrypoint=bash churn-prediction
gunicorn --bind=0.0.0.0:9696 predict_:app

# Alternative, run directly from image
docker run -it -p 9696:9696 churn-prediction:latest

### Running FastAPI app with Docker
docker build --tag fastapi-churn:latest .

docker run -it -p 8000:8000 fastapi-churn:latest 