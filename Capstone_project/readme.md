### Capstone Project 

A first project of MLZoomcamp. This repo is served with purpose of demonstration how far we fully utilise  
what we have learned until week 7.


#### Dataset Description

*Original dataset* []

This data was gathered from subjects/patients in 

### Attribute Information

```
* Age	Age of the patient	years	[40,…, 95]
* Anaemia	Decrease of red blood cells (haemoglobin) 	binary	0, 1
* Creatinine_phosphokinase	CPK level in the blood	mcg/L	[23, …, 7861]
* Diabetes	Presence of diabetes	binary	0, 1
* Ejection_fraction	Percentage of blood leaving the heart at each contraction	float	[14, …, 80]
* High Blood Pressure	Presence of high blood pressure	binary	0, 1
* Platelets	Platelets in the blood	Kiloplatelets/mL	[25.01, …, 850.00]
* Serum_creatinine	Creatinine level in the blood	mg/dl	[0.50, …, 9.4]
* Serum_sodium	Sodium level in the blood	mEq/L	[114, …, 148]
* Sex	Male or female 	binary	0, 1
* Smoking	Smoking or not smoking	binary	0, 1
* Time	Follow-up period	days	[4, …, 285]
* Death_event	Confirmed death during follow-up period	binary	0, 1
```
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



### Relevant Paper


#### Problem Context/Project Description

For this midterm project, a binary classification model is implemented on the case of heart failure with an aim of predicting patients' survival.

Models involved in learning and generalising the context are decision tree, random forest and XGBoost. XGBoost/random forest performed at the best on picking deceased patients from heart failure (precision)/ recognising deceased patients from all of retrieved records paired with true label of deceased (recall) while decision tree achieved a superiority in acquiring survived patients (NPV). 

A notebook with a detailed description of the EDA and model selection is presented in `EDA_model.ipynb`. Python scripts that specifically designed for training and storing its artifact are prepared in `training.py` and `training_bentoml.py`. A flask application served for responding to input data submitted from `send_data.py` is available in file `prediction_service.py`. The model registry is called from that flask for deployment (in waitress/gunicorn).


#### Methodology


#### Files

- `readme.md`
- `heart_failure.csv`
- `EDA_model.ipynb`
- `training.py`
- `training_bentoml.py`
- `prediction_flask.py`
- `prediction_service.py`
- `send_data.py`
- `Dockerfile`
- `Pipfile`
- `bentoml.yaml`



#### How to Run the Code

Pipenv is used for package dependency and virtual environment management. The code has been run on a local notebook with Windows and executed with bash Linux.

Please follow these steps to run the code.

    1) Prepare `Pipenv` on a directory containing any file you wish to work with. Follow these steps if you have not installed it before.
    2) Installing with command `pipenv install` will let the program to install modules you have included with.
    3) After installation with pipenv completes, activate pipenv environemtn with `pipenv shell`.
    4) Run `training.py` that generate prediction output ready for you to observe its predictive and generalization capability. Then, you can make a new collection of artifact of training results with command `python training_bentoml.py`.
    5) You will see a list of model stored in BentoML. You will need to use this artifact for later use. 
    6) Time to proceed to deployment of a trained model. Start flask app with command `waitress-serve --listen=0.0.0.0:9696 prediction_flask:app`, then open a new bash tab that lets you send a data with `python send_data.py`.
    7) You can test if the modelling artifact can be reused for prediction with `bentoml serve prediction_service:svc`.
    

    For stopping the program in 6) and 7), hit buttons `CTRL + C` on your keyboard.

After you're done, you may deactivate your pipenv environment with:

    `exit`




#### Deployment

##### Docker and Flask

##### BentoML


This structure takes an inspiration from 
https://github.com/ziritrion/ml-zoomcamp/tree/main/07_midterm_project