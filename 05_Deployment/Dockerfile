# Flask app
# FROM python:3.8.13-slim

# RUN pip install pipenv

# WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]

# RUN pipenv install --system --deploy

# COPY ["predict_.py", "model_C=2.0.bin", "./"]

# EXPOSE 9696

# ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_:app"]

# FastAPI app
FROM python:3.8.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_fa.py", "model_C=2.0.bin", "./"]

EXPOSE 8000

CMD ["uvicorn", "predict_fa:app", "--host", "0.0.0.0", "--port", "8000"]