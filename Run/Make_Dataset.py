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

pd.set_option('max_rows', 9999)

cat_cols = ["site_id", "building_id", "primary_use", "hour", "day", "weekday",
            "month", "meter", 'building_id_month', 'building_id_meter_month']

read_dtypes = {
    'building_id': 'uint16',
    'meter': 'uint8',
    'meter_reading': 'float32'
}

read_dtypes_building = {
    'building_id': 'uint16',
    'site_id': 'uint8',
    'square_feet': 'float32'
}

read_dtypes_weather = {
    'air_temperature': 'float32',
    'dew_temperature': 'float32',
    'wind_speed': 'float32',
    'wind_diraction': 'int32',
    'sea_level_pressure': 'float32'
}


def set_dtypes(df, cat_cols):
    # category
    for c in cat_cols:
        try:
            df[c] = df[c].astype('category')
        except:
            pass

    return df

# Config  #####################################################################
today = (datetime.datetime.now()).strftime('%Y%m%d')

# Prep Train Data  #####################################################################
print('Train...')
_start = time.time()
train = pd.read_csv("../input/train.csv", dtype=read_dtypes)
df_weather_train = pd.read_csv("../input/weather_train.csv", dtype=read_dtypes_weather)
df_building = pd.read_csv("../input/building_metadata.csv", dtype=read_dtypes_building)

# Prepare Train Data
Dataset = PreprocessingDataset()
Dataset.prep(train, df_weather_train, df_building, mode='train')
Dataset.df = set_dtypes(Dataset.df, cat_cols=cat_cols)

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
_ = model.train(Dataset.df.sample(frac=0.05), **g_params)

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
# Chunk_size ver.
print('Test...')
chunksize = 15000000
_start = time.time()
test_gen = pd.read_csv("../input/test.csv", chunksize=chunksize, dtype=read_dtypes)

# Prepare Test Data
for i, test in enumerate(test_gen):
    df_weather_test = pd.read_csv("../input/weather_test.csv", dtype=read_dtypes_weather)
    df_building = pd.read_csv("../input/building_metadata.csv", dtype=read_dtypes_building)
    test_num = 41697600
    limit = int(np.ceil(test_num / chunksize))
    print("\r" + str(i + 1) + "/" + str(limit), end="")
    sys.stdout.flush()

    Dataset.prep(test, df_weather_test, df_building, mode='test')
    Dataset.df = set_dtypes(Dataset.df, cat_cols=cat_cols)

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




# Prep test Data  #####################################################################
print('test...')
_start = time.time()
test = pd.read_csv("../input/test.csv", dtype=read_dtypes)
df_weather_test = pd.read_csv("../input/weather_test.csv", dtype=read_dtypes_weather)
df_building = pd.read_csv("../input/building_metadata.csv", dtype=read_dtypes_building)

# Prepare test Data
Dataset = PreprocessingDataset()
Dataset.prep(test, df_weather_test, df_building, mode='test')
Dataset.df = set_dtypes(Dataset.df, cat_cols=cat_cols)

# Memory Clear
del test, df_weather_test, df_building
gc.collect()

# Save Preprocessed test Data
with open(f'../input/prep_test_{today}.pkl', 'wb') as f:
    pickle.dump(Dataset, f, protocol=4)

print('Prep testData Shape: ', Dataset.df.shape)

elapsedtime = time.time() - _start
print('test Preprocessing Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')
