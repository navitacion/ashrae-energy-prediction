import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import sys

from sklearn.metrics import mean_squared_error

import lightgbm as lgb


class Trainer:

    def __init__(self):
        pass

    def train(self, df, params, cv, num_boost_round, early_stopping_rounds, verbose, split=None):
        self.y = np.log1p(df['meter_reading'])
        self.x = df.drop(['meter_reading'], axis=1)
        self.cv = cv
        self.oof = 0.0
        self.models = []
        self.features = self.x.columns

        if split is None:
            _cv = cv.split(self.x)
        else:
            _cv = cv.split(self.x, self.x[split])

        for i, (trn_idx, val_idx) in enumerate(_cv):
            print('Fold {} Model Creating...'.format(i + 1))
            _start = time.time()

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

            print('Fold {}: {:.5f}'.format(i + 1, error))

            elapsedtime = time.time() - _start
            print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))
            print('')

            self.models.append(model)
            del model
        print('OOF Error: {:.5f}'.format(self.oof))

        return self.models

    def predict(self, df, step_size=500):

        if 'row_id' in df.columns:
            df.drop('row_id', axis=1, inplace=True)

        if step_size is not None:
            i = 0
            res = []
            for j in range(int(np.ceil(df.shape[0] / step_size))):
                res.append(np.expm1(sum(
                    [model.predict(df.iloc[i:i + step_size], num_iteration=model.best_iteration) for model in
                     self.models]) / self.cv.n_splits))
                i += step_size

            res = np.concatenate(res)

        else:
            res = np.zeros(len(df))
            for model in self.models:
                res += model.predict(df) / self.cv.n_splits

        return res

    def get_feature_importance(self):
        importance = np.zeros(len(self.features))

        for i in range(len(self.models)):
            importance += self.models[i].feature_importance() / len(self.models)

        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        })
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        fig = plt.figure(figsize=(12, 20))
        sns.barplot(x='importance', y='feature', data=importance_df)
        today = datetime.datetime.now().strftime('%Y%m%d')
        plt.savefig('.../Output/FeatureImportance_{}.png'.format(today))
