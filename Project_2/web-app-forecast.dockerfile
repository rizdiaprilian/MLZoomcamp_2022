# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["pred_app.py", "ets_engine.py", "./"]

COPY ["Internet_sales_UK_preprocessed.csv", "./"]

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "pred_app.py", "--server.port=8501", "--server.address=0.0.0.0"]