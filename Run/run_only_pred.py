from Utils.utils import Dataloader, Trainer
from Utils.Parameter import *
import gc, pickle, datetime
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# Config  ####################################################################################
model_pkl_path = '../Model/trainer_lgb_20191129.pkl'
test_pkl_path = '../input/prep_test_20191129.pkl'

# Predict  ####################################################################################
with open(model_pkl_path, 'rb') as f:
    trainer = pickle.load(f)

with open(test_pkl_path, 'rb') as f:
    test_data = pickle.load(f)

trainer.predict(test_data)
