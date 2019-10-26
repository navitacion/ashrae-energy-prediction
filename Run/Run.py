import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

from Utils.Dataset import PreprocessingDataset
from Utils.Trainer import Trainer
from Utils.Parameter import *


# Config  #####################################################################
Sampling_rate = None
SEED = 42

# Prep Train Data  #####################################################################
print('Data Loading...')
train = pd.read_csv("../input/train.csv")
df_weather_train = pd.read_csv("../input/weather_train.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

# Sampling
if Sampling_rate is not None:
    train = train.sample(frac=Sampling_rate, random_state=SEED)

data = PreprocessingDataset()
data.prep(train, df_weather_train, df_building, mode='train')
del train, df_weather_train, df_building
print('Data Already...')

# Model Create  #####################################################################
model = Trainer()
_ = model.train(data.df, **g_params)

# Plot Feature Importances  #####################################################################
# model.get_feature_importance()

# Prediction  #####################################################################
# Chunksize ver

chunk_size = 50000
test_reader = pd.read_csv("../input/test.csv", chunksize=chunk_size)
df_weather_test = pd.read_csv("../input/weather_test.csv")
df_building = pd.read_csv("../input/building_metadata.csv")

pred_all = []

for test in tqdm(test_reader):
    data.prep(test, df_weather_test, df_building, mode='test')
    pred = model.predict(data.df, step_size=None)
    pred_all.append(pred)

pred_all = np.concatenate(pred_all)

# Make Submission File
sub = pd.read_csv("../input/sample_submission.csv")
sub["meter_reading"] = pred_all
today = datetime.datetime.now().strftime('%Y%m%d')
sub.to_csv("../Output/submission_{}_oof_{:.3f}.csv".format(today, model.oof), index=False)
