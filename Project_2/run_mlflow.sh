mlflow server --backend-store-uri file:///D:/github_repos/mlzoomcamp/MLZoomcamp_2022/Project_2/mlruns --no-serve-artifacts

## Build Docker Image for streamlit
docker build -t streamlit-forecast:v1  -f web-app-forecast.dockerfile .