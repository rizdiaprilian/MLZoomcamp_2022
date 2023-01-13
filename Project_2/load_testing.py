import os, sys
import numpy as np
import pandas as pd 
from ets_experiment import load_data, splitting_data

import pickle


with open(r"holt_winter_model.pickle", "rb") as input_file:
   model = pickle.load(input_file)

rs = model.forecast(55)
print(type(rs))
