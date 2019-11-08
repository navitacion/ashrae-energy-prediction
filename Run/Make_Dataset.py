import numpy as np
import pandas as pd
import datetime
import time
import sys
import gc
import pickle
import glob
from tqdm import tqdm

from Utils.Dataset import PreprocessingDataset
from Utils.Trainer import Trainer
from Utils.Parameter import *

today = (datetime.datetime.now()).strftime('%Y%m%d')
cat_cols = ["site_id", "building_id", "primary_use", "hour", "day", "weekday",
            "month", "meter", 'building_id_month', 'building_id_meter_month']

read_dtypes = {
    'meter': 'uint8',
    'meter_reading': 'float32'
}

read_dtypes_weather = {
    'air_temperature': 'float32',
    'dew_temperature': 'float32',
    'wind_speed': 'float32'
}

def set_dtypes(df, cat_cols):
    # float16
    cols = df.select_dtypes(include=[np.float64]).columns
    for c in cols:
        df[c] = df[c].astype(np.float32)
    # category
    for c in cat_cols:
        try:
            df[c] = df[c].astype('category')
        except:
            pass

    return df

# Prep Train Data  #####################################################################
print('Train...')
_start = time.time()
train = pd.read_csv("../input/train.csv", dtype=read_dtypes)
df_weather_train = pd.read_csv("../input/weather_train.csv", dtype=read_dtypes_weather)
df_building = pd.read_csv("../input/building_metadata.csv")

# train = train.sample(frac=0.01)

# Prepare Train Data
Dataset = PreprocessingDataset()
Dataset.prep(train, df_weather_train, df_building, mode='train')

# Data Type  #####################################################################
# Dataset.df = set_dtypes(Dataset.df, cat_cols)

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

# Check Importance  #####################################################################
# Model Create
model = Trainer()
_ = model.train(Dataset.df.sample(frac=0.04), **g_params)

# Plot Feature Importances
model.get_feature_importance()

del Dataset, model, _
gc.collect()

# Prep Test Data  #####################################################################
# Chunksize Ver.

# Load PKL File
with open(f'../input/prep_train_{today}.pkl', 'rb') as f:
    Dataset = pickle.load(f)

# Memory Clear
del Dataset.df
gc.collect()

# Create Test Dataset  #####################################################################
print('Test...')
chunksize = 15000000
_start = time.time()
test_gen = pd.read_csv("../input/test.csv", chunksize=chunksize, dtype=read_dtypes)

# Prepare Test Data
for i, test in enumerate(test_gen):
    df_weather_test = pd.read_csv("../input/weather_test.csv", dtype=read_dtypes_weather)
    df_building = pd.read_csv("../input/building_metadata.csv")
    test_num = 41697600
    limit = int(np.ceil(test_num / chunksize))
    print("\r" + str(i + 1) + "/" + str(limit), end="")
    sys.stdout.flush()

    Dataset.prep(test, df_weather_test, df_building, mode='test')

    # Data Type  #####################################################################
    Dataset.df = set_dtypes(Dataset.df, cat_cols)

    # Save Preprocessed Test Data  #####################################################################
    with open(f'../input/prep_test_{today}_{i}.pkl', 'wb') as f:
        pickle.dump(Dataset, f, protocol=4)

    # Memory Clear
    del df_weather_test, df_building
    gc.collect()

elapsedtime = time.time() - _start
print('Test Preprocessing Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')

print('Data Already...')
