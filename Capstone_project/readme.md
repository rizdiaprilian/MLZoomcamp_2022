### Capstone Project 

A first project of MLZoomcamp. This repo is served with purpose of demonstration how far we fully utilise  
what we have learned until week 7.


#### Dataset Description

*Original dataset* []

This data was gathered from subjects/patients in 

### Attribute Information

```



```
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
- `prediction_service.py`
- `send_data.py`
- `Dockerfile`
- `Pipfile`
- `bentoml.yaml`



#### Result and Analysis


#### Deployment

##### Docker and Flask

##### BentoML


This structure takes an inspiration from 
https://github.com/ziritrion/ml-zoomcamp/tree/main/07_midterm_project