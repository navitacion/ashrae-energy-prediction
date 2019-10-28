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
Sampling_rate = None
SEED = 42
chunk_size = 50000
today = datetime.datetime.now().strftime('%Y%m%d')

# Prep Train Data  #####################################################################
print('Data Loading...')
train = pd.read_csv("../input/train.csv")
df_weather_train = pd.read_csv("../input/weather_train.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

# Sampling
if Sampling_rate is not None:
    train = train.sample(frac=Sampling_rate, random_state=SEED)

# Prepare Train Data
Dataset = PreprocessingDataset()
Dataset.prep(train, df_weather_train, df_building, mode='train')
# Save Preprocessed Train Data
with open('../input/prep_train.pkl', 'wb') as f:
    pickle.dump(Dataset, f, protocol=4)
# Memory Clear
del train, df_weather_train, df_building
gc.collect()
print('Data Already...')

# Model Create  #####################################################################
model = Trainer()
_ = model.train(Dataset.df, **g_params)
# save models
with open(f'../Model/lgb_models_{today}.pkl', 'wb') as f:
    pickle.dump(model.models, f)

# Plot Feature Importances  #####################################################################
# model.get_feature_importance()

# Prediction  #####################################################################
print('Prediction')
_start = time.time()
df_weather_test = pd.read_csv("../input/weather_test.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

# Prepare Test Data
test = pd.read_csv("../input/test.csv")
Dataset.prep(test, df_weather_test, df_building, mode='test')
# Save Preprocessed Train Data
with open('../input/prep_test.pkl', 'wb') as f:
    pickle.dump(Dataset, f, protocol=4)
# Memory Clear
del test, df_weather_test, df_building
gc.collect()
pred_all = model.predict(Dataset.df, step_size=chunk_size)

# Make Submission File
sub = pd.read_csv("../input/sample_submission.csv")
sub["meter_reading"] = pred_all
sub.to_csv(f"../Output/submission_{today}_oof_{model.oof:.3f}.csv", index=False)

print('\nSubmit File Already!')
elapsedtime = time.time() - _start
print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')
