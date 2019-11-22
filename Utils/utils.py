import numpy as np
import pandas as pd
import gc, os
import datetime
import pickle
from functools import wraps
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from ngboost.distns import Normal, LogNormal

from ngboost import NGBRegressor

from .Preprocessing import reduce_mem_usage, prep_weather_data, prepare_data


class Dataloader:

    def __init__(self):
        self.input_dir = '../input'
        self.output_dir = '../Output'
        self.today = datetime.datetime.now().strftime('%Y%m%d')
        self.importance_df = None
        self.X = None
        self.y = None

    def prep(self, phase='train'):
        self.phase = phase
        path_core_data = None
        path_building = None
        path_weather_data = None

        # Set file path
        if phase == 'train' or phase == 'importance':
            path_core_data = os.path.join(self.input_dir, 'train.csv')
            path_building = os.path.join(self.input_dir, 'building_metadata.csv')
            path_weather_data = os.path.join(self.input_dir, 'weather_train.csv')

        elif phase == 'test':
            path_core_data = os.path.join(self.input_dir, 'test.csv')
            path_building = os.path.join(self.input_dir, 'building_metadata.csv')
            path_weather_data = os.path.join(self.input_dir, 'weather_test.csv')

        # Data Loading  ############################################
        print('Data Loading...')
        df = pd.read_csv(path_core_data)
        df = reduce_mem_usage(df, use_float16=True)

        building = pd.read_csv(path_building)
        building = reduce_mem_usage(building, use_float16=True)
        le = LabelEncoder()
        building.primary_use = le.fit_transform(building.primary_use)

        weather = pd.read_csv(path_weather_data)
        weather = reduce_mem_usage(weather, use_float16=True)
        weather = prep_weather_data(weather)

        # Feature Importance  ############################################
        # Prepare X, Y Data
        print('Preprocessing...')
        if phase == 'train':
            self.X, self.y = prepare_data(df, building, weather)

        elif phase == 'test':
            self.X, self.y = prepare_data(df, building, weather, test=True)

        elif phase == 'importance':
            self.X, self.y = prepare_data(df, building, weather, frac=0.01)

        for c in self.X.select_dtypes(np.float64).columns.tolist():
            self.X[c] = self.X[c].astype(np.float32)
        gc.collect()

        del df, building, weather
        gc.collect()

    def save_pickle(self):
        with open(os.path.join(self.output_dir, 'pycharm_prep_{}_{}.pkl'.format(self.phase, self.today)), 'wb') as f:
            pickle.dump([self.X, self.y], f)

    def get_feature_importance(self):
        print('Get Feature Importance...')
        assert self.phase == 'importance', 'Use Importance method'

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "num_leaves": 40,
            "learning_rate": 0.05,
            "feature_fraction": 0.85,
            "reg_lambda": 2,
            "metric": "rmse"
        }

        model = lgb.train(params, train_set=train_data, num_boost_round=1000,
                          valid_sets=[train_data, valid_data], verbose_eval=None, early_stopping_rounds=200)

        self.importance_df = pd.DataFrame({
            "feature": self.X.columns.values,
            "importance": model.feature_importance()
        })
        self.importance_df.sort_values(by='importance', ascending=False, inplace=True)
        self.importance_df.to_csv(f'../Importance/importance_{self.today}.csv', index=False)

        del X_train, X_val, y_train, y_val, train_data, valid_data, model
        gc.collect()
        print('DONE!')

        return self.importance_df


class Trainer:

    def __init__(self, model_type='lgb'):
        self.model_type = model_type
        self.X_train = None
        self.y_train = None
        self.features = None
        self.data = None

    def train_half(self, params, train_data, num_boost_round, early_stopping_rounds, verbose,
                   importance_df, use_feature_num=None):
        print('Model Creating...')
        self.data = train_data

        if use_feature_num is not None:
            self.features = importance_df['feature'][:use_feature_num].tolist()
        else:
            self.features = self.data.X.columns

        self.models = []
        assert self.data.phase == 'train', 'Use Train Dataset!'

        self.features = [c for c in self.features if c not in ['M']]

        self.X_train = self.data.X[self.features]
        self.y_train = self.data.y

        del self.data
        gc.collect()

        if self.model_type == 'lgb':
            print('LightGBM Model Creating...')
            d_half_1 = lgb.Dataset(self.X_train[:int(self.X_train.shape[0] / 2)],
                                   label=self.y_train[:int(self.X_train.shape[0] / 2)])
            d_half_2 = lgb.Dataset(self.X_train[int(self.X_train.shape[0] / 2):],
                                   label=self.y_train[int(self.X_train.shape[0] / 2):])

            print("Building model with first half and validating on second half:")
            model_1 = lgb.train(params, train_set=d_half_1, num_boost_round=num_boost_round,
                                valid_sets=[d_half_1, d_half_2], verbose_eval=verbose,
                                early_stopping_rounds=early_stopping_rounds)
            self.models.append(model_1)

            print('')
            print("Building model with second half and validating on first half:")
            model_2 = lgb.train(params, train_set=d_half_2, num_boost_round=num_boost_round,
                                valid_sets=[d_half_2, d_half_1], verbose_eval=verbose,
                                early_stopping_rounds=early_stopping_rounds)
            self.models.append(model_2)

        elif self.model_type == 'cat':
            print('CatBoost Model Creating...')
            cat_features_index = np.where(self.X_train.dtypes == 'category')[0]
            d_half_1 = Pool(self.X_train[:int(self.X_train.shape[0] / 2)],
                            label=self.y_train[:int(self.X_train.shape[0] / 2)],
                            cat_features=cat_features_index)
            d_half_2 = Pool(self.X_train[int(self.X_train.shape[0] / 2):],
                            label=self.y_train[int(self.X_train.shape[0] / 2):],
                            cat_features=cat_features_index)

            params['iterations'] = num_boost_round
            print("Building model with first half and validating on second half:")
            model_1 = CatBoostRegressor(**params)
            model_1.fit(d_half_1, eval_set=d_half_2, use_best_model=True,
                        early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            self.models.append(model_1)

            print('')
            print("Building model with second half and validating on first half:")
            model_2 = CatBoostRegressor(**params)
            model_2.fit(d_half_2, eval_set=d_half_1, use_best_model=True,
                        early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            self.models.append(model_2)

        elif self.model_type == 'ng':
            print('NGBoost Model Creating...')
            print("Building model with first half and validating on second half:")
            model_1 = NGBRegressor(Base=default_tree_learner, Dist=Normal,  # Normal, LogNormal
                                   Score=MLE(), natural_gradient=True, verbose=True,
                                   n_estimators=num_boost_round, verbose_eval=verbose,
                                   learning_rate=0.01, minibatch_frac=1.0)

            model_1.fit(self.X_train[:int(self.X_train.shape[0] / 2)],
                        self.y_train[:int(self.X_train.shape[0] / 2)],
                        X_val=self.X_train[int(self.X_train.shape[0] / 2):],
                        Y_val=self.y_train[int(self.X_train.shape[0] / 2):])
            self.models.append(model_1)

            print('')
            print("Building model with second half and validating on first half:")
            model_2 = NGBRegressor(Base=default_tree_learner, Dist=Normal,  # Normal, LogNormal
                                   Score=MLE(), natural_gradient=True, verbose=True,
                                   n_estimators=num_boost_round, verbose_eval=verbose,
                                   learning_rate=0.01, minibatch_frac=1.0)

            model_2.fit(self.X_train[int(self.X_train.shape[0] / 2):],
                        self.y_train[int(self.X_train.shape[0] / 2):],
                        X_val=self.X_train[:int(self.X_train.shape[0] / 2)],
                        Y_val=self.y_train[:int(self.X_train.shape[0] / 2)])
            self.models.append(model_2)

        del self.X_train, self.y_train
        gc.collect()

        return self.models

    def predict(self, test_data, model_list=None):
        print('Prediction...')
        assert test_data.phase == 'test', 'Use Test Dataset!'

        # Set Predict values
        pred = np.zeros(len(test_data.y))

        if model_list is None:
            model_list = self.models

        test_data.X = test_data.X[self.features]

        # Prediction
        for model in self.models:
            if self.model_type == 'lgb':
                pred += np.expm1(model.predict(test_data.X, num_iteration=model.best_iteration)) / int(len(model_list))
            elif self.model_type == 'cat' or self.model_type == 'ng':
                pred += np.expm1(model.predict(test_data.X)) / int(len(model_list))

        today = datetime.datetime.now().strftime('%Y%m%d')
        submission = pd.DataFrame({"row_id": test_data.y, "meter_reading": np.clip(pred, 0, a_max=None)})
        submission.to_csv(f"../Output/submission_from_nb_{self.model_type}_{today}.csv", index=False)
        print("DONE")
