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
_start = time.time()
chunk_size = 50000
today = datetime.datetime.now().strftime('%Y%m%d')
use_col_nums = 40

train_data_path = '../input/prep_train_20191030.pkl'
test_data_path = '../input/prep_test_20191030.pkl'
feature_importance_list_path = '../Importance/importance_20191030.csv'

# Prep Train Data  #####################################################################
set_cols = pd.read_csv(feature_importance_list_path)['feature'][:use_col_nums].tolist()
train_set_cols = set_cols + ['meter_reading']

# Load Pkl File  #####################################################################
print('Create Model...')
with open(train_data_path, 'rb') as f:
    Dataset = pickle.load(f)

# Model Create  #####################################################################
model = Trainer()
_ = model.train(Dataset.df[train_set_cols], **g_params)
# save models
with open(f'../Model/lgb_models_{today}.pkl', 'wb') as f:
    pickle.dump(model, f)

# Prediction  #####################################################################

# Load Pkl File  #####################################################################
print('Prediction...')
with open(test_data_path, 'rb') as f:
    Dataset = pickle.load(f)

# Pred  #####################################################################
pred_all = model.predict(Dataset.df[set_cols], step_size=chunk_size)

# Make Submission File  #####################################################################
sub = pd.read_csv("../input/sample_submission.csv")
sub["meter_reading"] = pred_all
sub.to_csv(f"../Output/submission_{today}_oof_{model.oof:.3f}.csv", index=False)

print('\nSubmit File Already!')
elapsedtime = time.time() - _start
print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
print('')
