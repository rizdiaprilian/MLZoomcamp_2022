### Capstone Project 

A first project of MLZoomcamp. This repo is served with purpose of demonstration how far we fully utilise  
what we have learned until week 7.


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

### Problem Context/Project Description

For this midterm project, a binary classification model is implemented on the case of heart failure with an aim of predicting patients' survival.

Models involved in learning and generalising the context are decision tree, random forest and XGBoost. XGBoost/random forest performed at the best on picking deceased patients from heart failure (precision)/ recognising deceased patients from all of retrieved records paired with true label of deceased (recall) while decision tree achieved a superiority in acquiring survived patients (NPV). 

A notebook with a detailed description of the Exploratory Data Analysis (EDA), and model building and tuning is presented in `project.ipynb`. Python scripts that specifically designed for training and storing its artifact are prepared in `train.py`. A flask application served for responding to input data submitted from `send_data.py` is available in file `prediction_service.py`. The model registry is called from that flask for deployment (in waitress/gunicorn).


### Files

- `readme.md`: A full description of the project for reader to gain a greater picture of this project.
- `heart_failure_clinical_records_dataset.csv`: The collection of heart failure records in CSV format.
- `project.ipynb` : A jupyter notebook containing model building and parameter fine-tuning. This file also build ML packages in bentoml.
- `heart_failure_profile.html`: A summary of exploratory data analysis on heart failure in web page. This was produced with `pandas_profiling`.
- `training.py`: A python app that build a bentoml registry.
- `prediction_service_sklearn.py`: A service app in sklearn that call a trained model from BentoML artifact to give a prediction to input data in flask service.
- `prediction_service_xgboost.py`: A service app in xgboost that call a trained model from BentoML artifact to give a prediction to input data in flask service.
- `send_data.py`: A python app that gives a request and delivers an input data to a service app to produce a prediction.
- `Pipfile`: A Pipfile for collection of libraries and modules dependencies.
- `bentoml.yaml`: A structured file to produce/build a ML service container.

#### EDA and Feature Importance

Analysis on heart failure dataset highlights a few findings to learn: 

    - All columns are completely free from missing values and type inconsistencies, thus ruling out requirements for data filling. 
    - Columns that holds binary data are in state of integer types. We convert them to categorical types with pandas map function.  
    - Visual graph sees non-gaussian (non-normal) distributions on features `creatinine_phosphokinase`, `platelets`, `serum_creatinine`, and `serum_sodium`. Since we use tree models in building predictive learning, transforming with np.log1p() or other functions is not necessary.
    - Numerical relationship shows strong correlation on target `DEATH_EVENT` to features `creatine`, `age`, `ejection_fraction`, and `time`.
    - Mutual information on categorical features shows an extremely weak relationship on target `DEATH_EVENT` to all categorical features.

#### How to Run the Code

    1) Prepare a file named `Pipfile` on a directory where projects locates. `Pipfile` gathers collection of modules which are utilized for development and production. 
    2) After gathering modules in `Pipfile`, install pipenv with command `pipenv install`. You can also install other modules if any update of requirements comes by (for example, `pipenv install xgboost` if we decide to include xgboost for adding another predictive modelling). 
    3) Activate pipenv environment with command `pipenv shell` following the completion of pipenv installation. At this point, we are ready to do a few tasks.
    4) Conduct data exploration and preprocessing before proceeding to model fitting with tree-based learning. Then, we tune parameters of three models that produce the best possible prediction on test set. Lastly, bentoml packages will be produced after execution finishes. These are extensively done in `project.ipynb`. 
    5) You will see a list of model stored in BentoML with `bentoml models list`.  
    6) Time to proceed to deployment of a trained model. Start a prediction service with command `bentoml serve prediction_service:svc`, then open a new bash tab that lets you send a data with `python send_data.py`. A response will appear a few seconds later. 
    7) Build bentoml docker package as specified in 'bentofile.yaml' with command `bentoml build`.
    8) Generate a docker container with `bentoml dockerize <service name:tag>` and it is ready to be deployed for production.

#### Performing under multiple requests in parallel

    1) Do the same as mentioned in previous section but with a little change: we will make multiple calls by running `locustfile.py`. Since this project is developed in Windows, so after executing `locust -H http://localhost:3000`, locust UI would appear in `http://localhost:8089`.
    2) Set number of users and spawn rate. Start with 100 and 10, respectively. Observe how the prediction service fare against concurrent requests: it would be great if responses run well without fails.  
    3) Reports are available to read in `Locust_async_reports` directory.



This structure takes an inspiration from 
https://github.com/ziritrion/ml-zoomcamp/tree/main/07_midterm_project