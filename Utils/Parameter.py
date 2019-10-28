from sklearn.model_selection import StratifiedKFold


# Model Params  ##################################################################
model_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'seed': 42
}


# General Params  ##################################################################

g_params = {
    'cv': StratifiedKFold(3, shuffle=True, random_state=42),
    'num_boost_round': 4000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'split': 'building_id',
    'params': model_params
}

