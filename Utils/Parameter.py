from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

# LightGBM
# Model Params  ##################################################################
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'seed': 42
}

lgb_params_2 = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 40,
    "learning_rate": 0.01,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
}

lgb_params_3 = {
    'objective': 'regression',
    'boosting_type': 'gbrt',
    'metric': 'rmse',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'max_bin': 255,
    'bagging_fraction': 0.42523401307460185,
    'bagging_freq': 6,
    'colsample_bytree': 0.7187743617504782,
    'feature_fraction': 0.8822412268247286,
    'max_depth': 10,
    'min_data_in_leaf': 42,
    'num_leaves': 279,
    'reg_lambda': 114.8060332041216,
    'subsample': 0.8631504025541011,
    'verbose': -1,
    'seed': 42
}



# CatBoost
# Model Params  ##################################################################
cat_params = {
    'learning_rate': 0.03,
    'eval_metric': 'RMSE',
    'loss_function': 'RMSE',
    'random_seed': 42,
    'metric_period': 10,
    'task_type': 'GPU',
    'depth': 8,
    }

