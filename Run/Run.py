from Utils.utils import Dataloader, Trainer
from Utils.Parameter import *
import gc, pickle, datetime
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# Config  ####################################################################################
num_boost_round = 10000
early_stopping_rounds = 500
verbose = 500
use_feature_num = 45
params = lgb_params_6
model_type = 'lgb'
Save_Pickle = False
today = datetime.datetime.now().strftime('%Y%m%d')

train_pkl_path = '../input/prep_train_20191129.pkl'
test_pkl_path = '../input/prep_test_20191129.pkl'
importance_path = '../Importance/importance_20191129.csv'

# Feature Importance  ####################################################################################
if Save_Pickle:
    data = Dataloader()
    data.prep(phase='importance')
    importance_df = data.get_feature_importance()
    del data
    gc.collect()
else:
    importance_df = pd.read_csv(importance_path)
print('')

# Train  ####################################################################################
print('Train Model')
if Save_Pickle:
    train_data = Dataloader()
    train_data.prep(phase='train')
    with open(f'../input/prep_train_{today}.pkl', 'wb') as f:
        pickle.dump(train_data, f, protocol=4)

    with open(f'../input/prep_train_{today}_list.pkl', 'wb') as f:
        pickle.dump([train_data.X, train_data.y], f, protocol=4)

else:
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)


trainer = Trainer(model_type=model_type)
model = trainer.train_half(params, train_data, num_boost_round, early_stopping_rounds, verbose,
                           importance_df=importance_df, use_feature_num=use_feature_num)
del train_data
gc.collect()

with open(f'../Model/trainer_{trainer.model_type}_{today}.pkl', 'wb') as f:
    pickle.dump(trainer, f, protocol=4)
print('')

# Predict  ####################################################################################
print('Prediction')
if Save_Pickle:
    test_data = Dataloader()
    test_data.prep(phase='test')
    with open(f'../input/prep_test_{today}.pkl', 'wb') as f:
        pickle.dump(test_data, f, protocol=4)
else:
    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

trainer.predict(test_data)
