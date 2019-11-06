import numpy as np
import pandas as pd
import gc
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


from .Preprocessing import prep_weather_data, prep_core_data, prep_building_data, reduce_mem_usage


class PreprocessingDataset:
    def __init__(self):
        self.df = None

    def prep(self, df, df_weather, df_building, mode='train'):

        # Core Data Prep  #####################################################################
        if mode == 'train':
            df = prep_core_data(df)

        # Weather Data Prep  #####################################################################
        df_weather = prep_weather_data(df_weather)

        # Building MetaData Prep  #####################################################################
        df_building = prep_building_data(df_building)

        # merge data  #####################################################################
        df = pd.merge(df, df_building, how="left", on=["building_id"])
        del df_building
        gc.collect()
        df = pd.merge(df, df_weather, how='left', on=["site_id", "timestamp"])
        self.df, _ = reduce_mem_usage(df)
        del df, df_weather
        gc.collect()

        # TargetEncoding  #####################################################################
        if mode == 'train':
            df_group = self.df.groupby('building_id')['meter_reading']
            self.building_mean = df_group.mean().astype(np.float16)
            self.building_median = df_group.median().astype(np.float16)
            self.building_min = df_group.min().astype(np.float16)
            self.building_max = df_group.max().astype(np.float16)
            self.building_std = df_group.std().astype(np.float16)

        self.df['building_mean'] = self.df['building_id'].map(self.building_mean)
        self.df['building_median'] = self.df['building_id'].map(self.building_median)
        self.df['building_min'] = self.df['building_id'].map(self.building_min)
        self.df['building_max'] = self.df['building_id'].map(self.building_max)
        self.df['building_std'] = self.df['building_id'].map(self.building_std)

        # Datetime  #####################################################################
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['month'] = self.df['timestamp'].dt.month.astype(np.uint8)
        self.df['day'] = self.df['timestamp'].dt.day.astype(np.uint8)
        self.df['hour'] = self.df['timestamp'].dt.hour.astype(np.uint8)
        self.df['weekday'] = self.df['timestamp'].dt.weekday.astype(np.uint8)

        # Holiday  #####################################################################
        dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
        self.df['is_holiday'] = (
            self.df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
        self.df.loc[(self.df['weekday'] == 5) | (self.df['weekday'] == 6), 'is_holiday'] = 1
        del us_holidays
        gc.collect()

        # Group feature  #####################################################################
        self.df['building_id_month'] = self.df['building_id'].astype(str) + '_' + self.df['month'].astype(str)
        self.df['building_id_meter_month'] = self.df['building_id'].astype(str) + '_' + \
                                             self.df['meter'].astype(str) + '_' + \
                                             self.df['month'].astype(str)

        self.df['building_id_meter_month_use'] = self.df['building_id'].astype(str) + '_' + \
                                             self.df['meter'].astype(str) + '_' + \
                                             self.df['month'].astype(str) + '_' + self.df['primary_use'].astype(str)

        # Frequency Encoding  #####################################################################
        cols = ['building_id', 'building_id_month', 'building_id_meter_month', 'building_id_meter_month_use']
        for col in cols:
            fq_encode = self.df[col].value_counts().to_dict()
            self.df[col + '_fq_enc'] = self.df[col].map(fq_encode)
            self.df[col + '_fq_enc'] = self.df[col + '_fq_enc'].astype(np.float16)

        # Sort Timestamp  #####################################################################
        if mode == 'train':
            self.df = self.df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        del self.df['timestamp']
        gc.collect()

        # LabelEncoder  #####################################################################
        list_cols = ['primary_use', 'building_id_month', 'building_id_meter_month', 'building_id_meter_month_use']
        if mode == 'train':
            self.ce_oe = ce.OrdinalEncoder(cols=list_cols, handle_unknown='impute')
            self.df = self.ce_oe.fit_transform(self.df)

        elif mode == 'test':
            self.df = self.ce_oe.transform(self.df)

        # Dropna    ####################################################################
        if mode == 'train':
            self.df.dropna(inplace=True)
