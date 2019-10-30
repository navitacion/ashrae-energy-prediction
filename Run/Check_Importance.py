import numpy as np
import pandas as pd
import datetime
import time
import sys
import gc
import pickle
from tqdm import tqdm

from Utils.Dataset import PreprocessingDataset
from Utils.Trainer import Trainer
from Utils.Parameter import *


# Config  #####################################################################
Sampling_rate = 0.05
SEED = 42
today = datetime.datetime.now().strftime('%Y%m%d')

# Prep Train Data  #####################################################################

# Load Pkl File  #####################################################################
print('Check Feature Importance...')
with open('../input/prep_train.pkl', 'rb') as f:
    Dataset = pickle.load(f)

# Sampling
if Sampling_rate is not None:
    Dataset.df = Dataset.df.sample(frac=Sampling_rate, random_state=SEED)

# Model Create  #####################################################################
model = Trainer()
_ = model.train(Dataset.df, **g_params)

# Plot Feature Importances  #####################################################################
model.get_feature_importance()
