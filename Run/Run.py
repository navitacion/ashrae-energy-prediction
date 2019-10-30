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
set_cols = pd.read_csv('../Importance/importance_20191029.csv')['feature'][:use_col_nums].tolist()
train_set_cols = set_cols + ['meter_reading']

# Prep Train Data  #####################################################################

# Load Pkl File  #####################################################################
print('Create Model...')
with open('../input/prep_train.pkl', 'rb') as f:
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
with open('../input/prep_test.pkl', 'rb') as f:
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
