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

today = datetime.datetime.now().strftime('%Y%m%d')

# # Prep Train Data  #####################################################################
print('Train...')
_start = time.time()
train = pd.read_csv("../input/train.csv")
df_weather_train = pd.read_csv("../input/weather_train.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

# Prepare Train Data
Dataset = PreprocessingDataset()
Dataset.prep(train, df_weather_train, df_building, mode='train')
# Memory Clear
del train, df_weather_train, df_building
gc.collect()
# Save Preprocessed Train Data
with open(f'../input/prep_train_{today}.pkl', 'wb') as f:
    pickle.dump(Dataset, f, protocol=4)

print('Prep TrainData Shape: ', Dataset.df.shape)

elapsedtime = time.time() - _start
print('Train Preprocessing Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')

# Memory Clear
del Dataset.df
gc.collect()

# Prep Test Data  #####################################################################
print('Test...')
_start = time.time()
test = pd.read_csv("../input/test.csv")
df_weather_test = pd.read_csv("../input/weather_test.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

# Prepare Test Data
Dataset.prep(test, df_weather_test, df_building, mode='test')
# Memory Clear
del test, df_weather_test, df_building
gc.collect()
# Save Preprocessed Test Data
with open(f'../input/prep_test_{today}.pkl', 'wb') as f:
    pickle.dump(Dataset, f, protocol=4)

print('Prep TestData Shape: ', Dataset.df.shape)

elapsedtime = time.time() - _start
print('Test Preprocessing Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')

print('Data Already...')
