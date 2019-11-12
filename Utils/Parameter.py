from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

# LightGBM
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

model_params_2 = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 40,
    "learning_rate": 0.01,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
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

g_params_3 = {
    'cv': KFold(n_splits=3, shuffle=False),
    'num_boost_round': 6000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'params': model_params
}

# CatBoost
# Model Params  ##################################################################
model_params_cat = {
        'learning_rate': 0.3,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'random_seed': 42,
        'metric_period': 10,
        'task_type': 'CPU',
        'depth': 8,
    }

# General Params  ##################################################################
g_params_cat = {
    'cv': StratifiedKFold(4, shuffle=True, random_state=42),
    'num_boost_round': 6000,
    'early_stopping_rounds': 100,
    'verbose': 1000,
    'split': 'building_id',
    'params': model_params_cat
}

g_params_cat_2 = {
    'cv': GroupKFold(4),
    'num_boost_round': 100,
    'early_stopping_rounds': 10,
    'verbose': 10,
    'group': 'month',
    'params': model_params_cat
}