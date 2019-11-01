from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


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
    'cv': StratifiedKFold(4, shuffle=True, random_state=42),
    'num_boost_round': 6000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'split': 'building_id',
    'params': model_params
}

g_params_by_col = {
    'cv': KFold(3, shuffle=True, random_state=42),
    'num_boost_round': 10000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'params': model_params
}


g_params_2 = {
    'cv': GroupKFold(4),
    'num_boost_round': 6000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'group': 'building_id_month',
    'params': model_params
}
