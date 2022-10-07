FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

# COPY ["model2.bin", "dv.bin", "./"]

# Install all the dependencies form the Pipenv file
COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# Copy your Flask script
COPY ["predict_hw_flask.py", "./"]

EXPOSE 9696

# Run it with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_hw_flask:app"]
