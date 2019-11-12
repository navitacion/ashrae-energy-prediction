import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import sys
import gc

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool


class Trainer:

    def __init__(self, model_type='lgb'):
        self.model_type = model_type

    def train(self, df, params, cv, num_boost_round, early_stopping_rounds, verbose, split=None, group=None):
        self.y = df['meter_reading']
        self.x = df.drop(['meter_reading'], axis=1)
        self.cv = cv
        self.oof = 0.0
        self.models = []
        self.features = self.x.columns

        if split is not None:
            _cv = cv.split(self.x, self.x[split])
        elif group is not None:
            _cv = cv.split(self.x, self.y, self.x[group])
        else:
            _cv = cv.split(self.x)

        for i, (trn_idx, val_idx) in enumerate(_cv):
            print('Fold {} Model Creating...'.format(i + 1))
            _start = time.time()

            if self.model_type == 'lgb':
                train_data = lgb.Dataset(self.x.iloc[trn_idx], label=self.y.iloc[trn_idx])
                val_data = lgb.Dataset(self.x.iloc[val_idx], label=self.y.iloc[val_idx], reference=train_data)

                model = lgb.train(params,
                                  train_data,
                                  num_boost_round=num_boost_round,
                                  valid_sets=(train_data, val_data),
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose)

                y_pred = model.predict(self.x.iloc[val_idx], num_iteration=model.best_iteration)
                error = np.sqrt(mean_squared_error(y_pred, self.y.iloc[val_idx]))
                self.oof += error / cv.n_splits
                self.models.append(model)

                print('Fold {}: {:.5f}'.format(i + 1, error))

            elif self.model_type == 'cat':
                cat_features_index = np.where(self.x.dtypes == 'category')[0]
                train_data = Pool(self.x.iloc[trn_idx], label=self.y.iloc[trn_idx], cat_features=cat_features_index)
                val_data = Pool(self.x.iloc[val_idx], label=self.y.iloc[val_idx], cat_features=cat_features_index)
                params['iterations'] = num_boost_round
                model = CatBoostRegressor(**params)
                model.fit(train_data,
                          eval_set=val_data,
                          use_best_model=True,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose=verbose)
                y_pred = model.predict(self.x.iloc[val_idx])
                error = np.sqrt(mean_squared_error(y_pred, self.y.iloc[val_idx]))
                self.oof += error / cv.n_splits
                self.models.append(model)

                print('Fold {}: {:.5f}'.format(i + 1, error))

            elapsedtime = time.time() - _start
            print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
            print('')

        print('OOF Error: {:.5f}'.format(self.oof))

        return self.models

    def train_half(self, df, params, cv, num_boost_round, early_stopping_rounds, verbose):
        self.y = df['meter_reading']
        self.x = df.drop(['meter_reading'], axis=1)
        self.cv = cv
        self.oof = 0.0
        self.models = []
        self.features = self.x.columns

        X_half_1 = self.x.iloc[:int(self.x.shape[0] / 2)]
        X_half_2 = self.x.iloc[int(self.x.shape[0] / 2):]

        y_half_1 = self.y.iloc[:int(self.x.shape[0] / 2)]
        y_half_2 = self.y.iloc[int(self.x.shape[0] / 2):]

        d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, free_raw_data=False)
        d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, free_raw_data=False)

        watchlist_1 = [d_half_1, d_half_2]
        watchlist_2 = [d_half_2, d_half_1]

        model_half_1 = lgb.train(params,
                                 train_set=d_half_1,
                                 num_boost_round=num_boost_round,
                                 valid_sets=watchlist_1,
                                 verbose_eval=verbose,
                                 early_stopping_rounds=early_stopping_rounds)

        self.models.append(model_half_1)
        oof = model_half_1.predict(X_half_2, num_iteration=model_half_1.best_iteration)
        rmse_1 = np.sqrt(mean_squared_error(oof, y_half_2))
        print('Half_1:  RMSE: {:.4f}'.format(rmse_1))

        model_half_2 = lgb.train(params,
                                 train_set=d_half_2,
                                 num_boost_round=num_boost_round,
                                 valid_sets=watchlist_2,
                                 verbose_eval=verbose,
                                 early_stopping_rounds=early_stopping_rounds)

        self.models.append(model_half_2)
        oof = model_half_2.predict(X_half_1, num_iteration=model_half_2.best_iteration)
        rmse_2 = np.sqrt(mean_squared_error(oof, y_half_1))
        print('Half_2:  RMSE: {:.4f}'.format(rmse_2))

        self.oof = (rmse_1 + rmse_2) / 2
        print('OOF Error: {:.5f}'.format(self.oof))

        return self.models

    def train_by_col(self, df, params, cv, num_boost_round, early_stopping_rounds, verbose, div_col, split=None):
        self.cv = cv
        self.models = {}
        self.features = [c for c in df.columns if c not in ['meter_reading']]
        self.oof = np.zeros(len(df))

        for col in df[div_col].unique().tolist():
            print(f'{div_col} : {col}')
            model_list = []
            temp = df[df[div_col] == col]
            df_index = temp.index
            self.y = temp['meter_reading']
            self.x = temp.drop(['meter_reading'], axis=1)
            del temp
            gc.collect()

            if split is None:
                _cv = cv.split(self.x)
            else:
                _cv = cv.split(self.x, self.x[split])

            for i, (trn_idx, val_idx) in enumerate(_cv):
                print('Fold {} Model Creating...'.format(i + 1))
                _start = time.time()

                if self.model_type == 'lgb':
                    train_data = lgb.Dataset(self.x.iloc[trn_idx], label=self.y.iloc[trn_idx])
                    val_data = lgb.Dataset(self.x.iloc[val_idx], label=self.y.iloc[val_idx], reference=train_data)

                    model = lgb.train(params,
                                      train_data,
                                      num_boost_round=num_boost_round,
                                      valid_sets=(train_data, val_data),
                                      early_stopping_rounds=early_stopping_rounds,
                                      verbose_eval=verbose)

                    self.oof[df_index[val_idx]] = model.predict(self.x.iloc[val_idx], num_iteration=model.best_iteration)

                    elapsedtime = time.time() - _start
                    print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
                    print('')

                    model_list.append(model)
                    del model

                elif self.model_type == 'cat':
                    cat_features_index = np.where(self.x.df.dtypes == 'category')[0]
                    train_data = Pool(self.x.iloc[trn_idx], label=self.y.iloc[trn_idx], cat_features=cat_features_index)
                    val_data = Pool(self.x.iloc[val_idx], label=self.y.iloc[val_idx], cat_features=cat_features_index)
                    model = CatBoostRegressor(**params.update({'iterations': num_boost_round}))
                    model.fit(train_data,
                              eval_set=val_data,
                              use_best_model=True,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose=verbose)
                    self.oof[df_index[val_idx]] = model.predict(self.x.iloc[val_idx])
                    elapsedtime = time.time() - _start
                    print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
                    print('')

            self.models.update({col: model_list})

        self.oof = np.sqrt(mean_squared_error(self.oof, df['meter_reading']))
        print('OOF Error: {:.5f}'.format(self.oof))

        return self.models

    def predict(self, df, step_size=500):

        i = 0
        res = []
        for j in range(int(np.ceil(df.shape[0] / step_size))):
            test_num = len(df)
            limit = int(np.ceil(test_num / step_size))
            print("\r" + str(j+1) + "/" + str(limit), end="")
            sys.stdout.flush()

            if self.model_type == 'lgb':
                res.append(np.expm1(sum(
                    [model.predict(df.iloc[i:i + step_size], num_iteration=model.best_iteration) for model in
                     self.models]) / self.cv.n_splits))
            elif self.model_type == 'cat':
                res.append(np.expm1(sum(
                    [model.predict(df.iloc[i:i + step_size]) for model in self.models]) / self.cv.n_splits))
            i += step_size

        res = np.concatenate(res)

        return res

    def predict_by_col(self, df, div_col, step_size=500):

        res = []
        row_id = []

        for col in df[div_col].unique().tolist():
            print(f'\n{div_col} : {col}')
            temp = df[df[div_col] == col]
            i = 0
            for j in range(int(np.ceil(temp.shape[0] / step_size))):
                test_num = len(temp)
                limit = int(np.ceil(test_num / step_size))
                print("\r" + str(j+1) + "/" + str(limit), end="")
                sys.stdout.flush()

                _temp = temp.iloc[i:i + step_size]
                if self.model_type == 'lgb':
                    res.append(np.expm1(sum([model.predict(_temp.drop('row_id', axis=1), num_iteration=model.best_iteration)
                                         for model in self.models[col]]) / self.cv.n_splits))
                elif self.model_type == 'cat':
                    res.append(
                        np.expm1(sum([model.predict(_temp.drop('row_id', axis=1))
                                      for model in self.models[col]]) / self.cv.n_splits))
                row_id.append(_temp['row_id'].values)

                i += step_size

        res = np.concatenate(res)
        row_id = np.concatenate(row_id)

        sub = pd.DataFrame({
            'row_id': row_id,
            'meter_reading': res
        })

        return sub

    def get_feature_importance(self):
        importance = np.zeros(len(self.features))

        for i in range(len(self.models)):
            
            if self.model_type == 'lgb':
                importance += self.models[i].feature_importance() / len(self.models)
            elif self.model_type == 'cat':
                importance += self.models[i].get_feature_importance() / len(self.models)

        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        })
        importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        today = datetime.datetime.now().strftime('%Y%m%d')
        importance_df.to_csv('../Importance/importance_{}.csv'.format(today), index=False)

        fig = plt.figure(figsize=(12, 20))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.savefig('../Importance/FeatureImportance_{}.png'.format(today))
