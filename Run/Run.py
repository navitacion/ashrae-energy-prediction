import numpy as np
import pandas as pd
import datetime
import time
import sys
import gc
import pickle
from tqdm import tqdm
import glob

from Utils.Dataset import PreprocessingDataset
from Utils.Trainer import Trainer
from Utils.Parameter import *


# Config  #####################################################################
_start = time.time()
chunk_size = 200000
today = datetime.datetime.now().strftime('%Y%m%d')
use_col_nums = 40
params = g_params

train_data_path = '../input/prep_train_20191105.pkl'
test_data_path = '../input/prep_test_20191105_*.pkl'
feature_importance_list_path = '../Importance/importance_20191105.csv'

# Prep Train Data  #####################################################################
set_cols = pd.read_csv(feature_importance_list_path)['feature'][:use_col_nums].tolist()
train_set_cols = set_cols + ['meter_reading']

# Load Pkl File  #####################################################################
# print('Create Model...')
# with open(train_data_path, 'rb') as f:
#     Dataset = pickle.load(f)
#
# Dataset.df['building_id_month'] = Dataset.df['building_id_month'].astype('category')

# Model Create  #####################################################################
# model = Trainer()
# _ = model.train(Dataset.df[train_set_cols], **params)
# # save models
# with open(f'../Model/lgb_models_{today}.pkl', 'wb') as f:
#     pickle.dump(model, f)

# Prediction  #####################################################################

with open(f'../Model/lgb_models_{today}.pkl', 'rb') as f:
    model = pickle.load(f)

model.model_type = 'lgb'

# Load Pkl File  #####################################################################
print('Prediction...')
id_ = []
pred = []

for testpath in glob.glob(test_data_path):
    with open(testpath, 'rb') as f:
        Dataset = pickle.load(f)

    id_.append(Dataset.df['row_id'].values)
    # Predict
    pred.append(model.predict(Dataset.df[set_cols], step_size=chunk_size))

    del Dataset
    gc.collect()

sub = pd.DataFrame({
    'row_id': np.concatenate(id_),
    'meter_reading': np.concatenate(pred)
})

# Create Submit DataFrame
sub.sort_values(by='row_id', ascending=True, inplace=True)
sub.to_csv(f"../Output/submission_{model.model_type}_{today}_oof_{model.oof:.3f}.csv", index=False)

print('\nSubmit File Already!')
elapsedtime = time.time() - _start
print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')
