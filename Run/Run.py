from Utils.utils import Dataloader, Trainer
from Utils.Parameter import *
import gc, pickle, datetime

import warnings
warnings.filterwarnings('ignore')


# Config  ####################################################################################
num_boost_round = 10000
early_stopping_rounds = 500
verbose = 500
use_feature_num = 40
params = lgb_params_3
model_type = 'lgb'
Save_Pickle = True
today = datetime.datetime.now().strftime('%Y%m%d')

# Feature Importance  ####################################################################################
data = Dataloader()
data.prep(phase='importance')
importance_df = data.get_feature_importance()
del data
gc.collect()
print('')

# Train  ####################################################################################
print('Train Model')
train_data = Dataloader()
train_data.prep(phase='train')
if Save_Pickle:
    with open(f'../input/prep_train_{today}.pkl', 'wb') as f:
        pickle.dump(train_data, f, protocol=4)

trainer = Trainer(model_type=model_type)
model = trainer.train_half(params, train_data, num_boost_round, early_stopping_rounds, verbose,
                           importance_df=importance_df, use_feature_num=use_feature_num)
del train_data
gc.collect()

if Save_Pickle:
    with open(f'../input/trainer_{trainer.model_type}_{today}.pkl', 'wb') as f:
        pickle.dump(trainer, f, protocol=4)
print('')

# Predict  ####################################################################################
print('Prediction')
test_data = Dataloader()
test_data.prep(phase='test')
if Save_Pickle:
    with open(f'../input/prep_test_{today}.pkl', 'wb') as f:
        pickle.dump(test_data, f, protocol=4)
trainer.predict(test_data)
