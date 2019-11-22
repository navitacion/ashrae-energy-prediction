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

# CatBoost
# Model Params  ##################################################################
cat_params = {
    'learning_rate': 0.3,
    'eval_metric': 'RMSE',
    'loss_function': 'RMSE',
    'random_seed': 42,
    'metric_period': 10,
    'task_type': 'GPU',
    'depth': 8,
    }

