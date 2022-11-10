## Capstone Project 

A first project of MLZoomcamp. This repo is served with purpose of demonstration how far we utilise what we have learned until week 7.


### Dataset Description

This project will take a focus on heart failure using a dataset collected from [Kaggle 2020](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) and compiled by Ahmad and colleagues (2017). The dataset holds the medical records of 299 patients aged between 40 and 95 years old accounted by 105 females and 194 males. This gathering was carried out at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad, Pakistan between April and December 2015. 

### Attribute Information

| Feature Name        | Explanation           | Measurement  |  Range  |
| ------------- | ------------- |  ----- |  ----- | 
| Age      | Age of the patient | years |  [40,…, 95]  |
| Anaemia      | Decrease of red blood cells (haemoglobin) |  binary |  0, 1  |
| Creatinine_phosphokinase | CPK level in the blood |   mcg/L | [23, …, 7861] |
|  Diabetes	 | Presence of diabetes |	binary |	0, 1 |
| Ejection_fraction | 	Percentage of blood leaving the heart at each contraction | 	float	| [14, …, 80] | 
| High Blood Pressure | 	Presence of high blood pressure | 	binary	| 0, 1 | 
| Platelets | 	Platelets in the blood | 	Kiloplatelets/mL	| [25.01, …, 850.00] | 
| Serum_creatinine | 	Creatinine level in the blood | 	mg/dl	| [0.50, …, 9.4] | 
| Serum_sodium |	Sodium level in the blood |	mEq/L |	[114, …, 148] |
| Sex |	Male or female | 	binary |	0, 1 |
| Smoking |	Smoking or not smoking |	binary |	0, 1 |
| Time |	Follow-up period |	days |	[4, …, 285] |
| Death_event |	Confirmed death during follow-up period |	binary |	0, 1 |



#### Relevant Paper
Chicco, D. and Jurman, G. (2020) ‘Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone’, BMC Medical Informatics and Decision Making, 20(1), pp. 1-16. Available at: https://doi.org/10.1186/s12911-020-1023-5.

Ahmad, T. et al. (2017) ‘Survival Analysis of Heart Failure Patients: A Case Study’, PLoS ONE, 12(7). Available at: https://doi.org/10.1371/journal.pone.0181001.

## 1) Description
### Problem Context/Project Description

For this midterm project, a binary classification model is implemented on the case of heart failure with an aim of predicting patients' survival.

Models involved in learning and generalising the context are decision tree, random forest and XGBoost. A Metric used to assess prediction capabilities is ROC-AUC.

A notebook with a detailed description of the Exploratory Data Analysis (EDA), and model building and tuning is presented in `project.ipynb`. Python scripts that specifically designed for training and storing its artifact are prepared in `train.py`. A prediction service composed of runners and APIs is served for responding to input data submitted from `send_data.py`: it is available in file `prediction_service_sklearn.py` and `prediction_service_xgboost.py`. An object service object is provided with a decorator method `svc.api` for defining API. A saved model in bentoml lists is retrieved for runner instances (in waitress/gunicorn).


### Files

- `readme.md`: A full description of the project for reader to gain a greater picture of this project.
- `heart_failure_clinical_records_dataset.csv`: The collection of heart failure records in CSV format.
- `project.ipynb` : A jupyter notebook containing model building and parameter fine-tuning. This file also build ML packages in bentoml.
- `heart_failure_profile.html`: A summary of exploratory data analysis on heart failure in web page. This was produced with `pandas_profiling`.
- `train.py`: A python app that build a bentoml registry.
- `prediction_service_sklearn.py`: A service app in sklearn that call a trained model from BentoML artifact to give a prediction to input data in flask service.
- `prediction_service_xgboost.py`: A service app in xgboost that call a trained model from BentoML artifact to give a prediction to input data in flask service.
- `send_data.py`: A python app that gives a request and delivers an input data to a service app to produce a prediction.
- `Pipfile`: A Pipfile for collection of libraries and modules dependencies.
- `bentoml.yaml`: A structured file to produce/build a ML service container.
- `requirements.txt`: A file for building a conda env.
- `locustfile.py`: A file for testing responsive predictions againts concurrent requests.


### Important Note about environment

This project used two different environments: 

- a conda using python 3.8.13 for working on `project.ipynb`. Since a module `Hyperopt` does not work in jupyter lab called from `Pipfile`, I choose to do this task on my environment that has been existing since a year ago. 
- a pipenv using python 3.9.13 for training, building bentos, testing services, and containerizing.

## 2) EDA, Feature Correlation, Feature Importance

A few findings to learn: 

**EDA**
- All columns are completely free from missing values and type inconsistencies, thus ruling out requirements for data filling. 
- Columns that holds binary data are in the state of integer types. We convert them to boolean/categorical types with pandas map function.
- Visual graph sees non-gaussian (non-normal) distributions on features `creatinine_phosphokinase`, `platelets`, `serum_creatinine`, and `serum_sodium`. Since we use tree models in building predictive learning, transforming with `np.log1p()` or other functions is not necessary.

**Group Risk Factor by Mean**
- Patients suffering `anaemia` tend to have higher risk than those who do not: the risk of mortality is 1.11 for anameia group against 0.916 for all patient not being treated as to having anaemia.
- Risk is higher for everyone that suffered from `high_blood_pressure`, in which its marginal proportion is 1.15. Meanwhile, people whose free from this ailment have lower risk (0.916).
- Risk differences due to `smoking` and `diabetes` are considerably lower than the group which are not subjected to these conditions.

**Mutual Information**
- Mutual information shows a similar result on feature imporatance: `anaemia` and `high_blood_pressure` are among categorical features that affect the risk of death. 
- Mutual information on categorical features shows an extremely weak relationship on target `DEATH_EVENT` to all categorical features.

**Feature Correlation**  
- A considerably high relationship on `DEATH_EVENT` with features `age`, `ejection_fraction`, `serum_creatinine`, and `time`.
    

## 3) Running the Code 

## Preparing Enviroments

1) Create a conda environment with command `conda create --name <env_name> --file requirements.txt`.
2) Prepare a file named `Pipfile` on a directory where projects locates. `Pipfile` gathers collection of modules which are utilized for development and production. 
3) After gathering modules in `Pipfile`, install pipenv with command `pipenv install`. You can also install other modules if any update of requirements comes by (for example, `pipenv install xgboost` if we decide to include xgboost for adding another predictive modelling). 

### Model Building and Fine-Tuning

1) Activate pipenv environment with command `pipenv shell` following the completion of pipenv installation. At this point, we are ready to do a few tasks.
2) Conduct data exploration and preprocessing before proceeding to model fitting with tree-based learning. Then, we build machine learning models followed by parameter tuning for three models that produce the best possible prediction on the test set. These are extensively done in `project.ipynb`. 

### Running Model Training and Results on the Test set

Run file `train.py` with additional arguments listed below:

    - `python train.py "Decision Tree" "sklearn"`
    - `python train.py "Random Forest" "sklearn"`
    - `python train.py "XGBoost" "xgboost"`

You will see a list of models are already stored in BentoML with `bentoml models list`. 

![image](screenshots/bentoml_models_list.png)

Model prediction results:

| Model       | ROC AUC           |
| ------------- | ------------- |  
| Decision Tree Classifier      | 0.801 |    
| Random Forest Classifier      | 0.867 | 
| XGBoost Tree-based | 0.889 |  

## 4) Testing BentoML

Start a prediction service with command `bentoml serve prediction_service_sklearn:svc` or `bentoml serve prediction_service_xgboost:svc`, then open a new bash tab that lets you send a data with `python send_data.py`. A response will appear a few seconds later. 

### Containerizing bentoml package into Docker Image

Specify lists of libraries and include a prediction service python script in file `bentofile.yaml`. Command `bentoml build` in a directory storing `bentofile.yaml` and `train.py` will allow a bentoml package to be generated.

![image](screenshots/ubuntu_ls.png)

![image](screenshots/bentoml_list.png)

Generate a docker container with `bentoml dockerize heart_failure_classifier:<tag>`. Then, run the container in local machine with `docker run -it --rm -p 3000:3000 <heart_failure_classifier:<tag>` and send a data by executing `python send_data.py` to see whether the container is successful in responding to its json input request.


#### Performing under multiple requests in parallel

1) Do the same as mentioned in previous section but with a little change: we will make multiple calls by running `locustfile.py`. Since this project is developed in Windows, so after executing `locust -H http://localhost:3000`, locust UI would appear in `http://localhost:8089`.
2) Set number of users and spawn rate. Start with 100 and 10, respectively. Observe how the prediction service fare against concurrent requests: it would be great if responses run well without fails.  
3) Reports are available to read in `Locust_async_reports` directory.

## Storing Docker Images to Docker Hub

In order to push image in docker hub, you must make a registration first. After it finishes, then make a repository in which an image generated by bentoml would be stored there. You can read more details [here](https://docs.docker.com/docker-hub/)

![image](screenshots/docker_push.png)

Choose one of images after seeing lists of docker images. For example:

![image](screenshots/docker_image_ls.png)

Take image ID and tag to create a new tag with command `docker tag <Image ID> 21492rar/heart_failure_machine_learning:<Tag>`. `21492rar/heart_failure_machine_learning` is the repository that I personally used for this project.

Then, execute command `docker push 21492rar/heart_failure_machine_learning:<Tag>` to begin image push to docker hub. If it succeeds, the repository will show collections of images.

![image](screenshots/docker_hub.png)

Docker images are available on my docker hub repository: https://hub.docker.com/r/21492rar/heart_failure_machine_learning/tags

## 5) Deploy to Google Cloud

This time, I use my existing remote host from Google Cloud. What I have to do:

1) Starting Compute Engine and preparation:
    - Update python in anaconda to 3.9.12
    - Create a new directory `mlzoomcamp` just to work with pipenv environment.
    - Authenticate gcloud with `gcloud auth login`. Just continue to authenticate with your personal account by pressing `Y`. As its first response you may see like this:

![image](screenshots/gcloud_auth_login1.png)

Then, copy that long code and switch window from remote host to local machine and paste it on bash terminal. A web browser will automatically start to load the page you need to enter your username and password. If this succeeds, the second response is given as shown below:

![image](screenshots/gcloud_auth_login2.png)

Copy that response code and back to the remote host, then paste it. After your credential is accepted, you just only need to choose your project ID (if any) and begin your work.

2) Configure container registry (https://cloud.google.com/container-registry/docs/advanced-authentication#gcloud-helper):
    - Configure Docker (if Docker has been installed) with `gcloud auth configure-docker`. A credential is saved in your user home directory

3) Download a docker image from my docker hub. Retrieve an image by running `docker pull 21492rar/heart_failure_machine_learning:g4ls2g266cfa27fs`.

4) Push a downloaded image to container registry in two steps:
    - `docker tag <IMAGE ID> <HOST NAME>/<PROJECT_ID>/<TARGET_IMAGE>:<TAG>`
    - `docker push <HOST NAME>/<PROJECT_ID>/<TARGET_IMAGE>:<TAG>`

![image](screenshots/container_registry.png)

![image](screenshots/container_registry2.png)

5) Deployment with Google Cloud Run (https://cloud.google.com/sdk/gcloud/reference/run/deploy):
    - Deploy the image stored in container registry with command `gcloud run deploy ml-serve --image=eu.gcr.io/data-eng-camp-apr22/heart_failure_service@sha256:be6bcf45bed2616a69be9a59112272ef277d70aa69229dfc4661bb4423067b12 --region=europe-west2 --port=3000`

![image](screenshots/copy_image_detail.png)

    - When it is successful, an unique url will appear. Copy the url to the browser of your choice. A service URL as a cloud deployment result: https://ml-serve-iy2jfge65q-nw.a.run.app

![image](screenshots/deploy_success.png)

![image](screenshots/deploy_success2.png)

    - You are ready to try input the data and test its predictive ability. JSON input which is written to the request body should be as shown in an example below:

```
{
    "age": 41.0,
    "anaemia": "no",
    "creatinine_phosphokinase": 213,
    "diabetes": "no",
    "ejection_fraction": 89,
    "high_blood_pressure": "no",
    "platelets": 234100.0,
    "serum_creatinine": 5.6,
    "serum_sodium": 120,
    "sex": "male",
    "smoking": "no",
    "time": 23
}
```

![image](screenshots/test_deploy_cloud.png)

    - The details of service health is presented shown below. 


![image](screenshots/cloud_run.png)


![image](screenshots/cloud_run2.png)

This structure takes an inspiration from 
https://github.com/ziritrion/ml-zoomcamp/tree/main/07_midterm_project