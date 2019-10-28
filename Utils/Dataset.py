import numpy as np
import pandas as pd
import gc
import category_encoders as ce


from .Preprocessing import prep_weather_data, prep_core_data, prep_building_data, reduce_mem_usage


class PreprocessingDataset:
    def __init__(self):
        self.df = None

    def prep(self, df, df_weather, df_building, mode='train'):

        # Core Data Prep  #####################################################################
        if mode == 'train':
            df = prep_core_data(df)

            df_group = df.groupby('building_id')['meter_reading']
            self.building_mean = df_group.mean().astype(np.float16)
            self.building_median = df_group.median().astype(np.float16)
            self.building_min = df_group.min().astype(np.float16)
            self.building_max = df_group.max().astype(np.float16)
            self.building_std = df_group.std().astype(np.float16)

        df['building_mean'] = df['building_id'].map(self.building_mean)
        df['building_median'] = df['building_id'].map(self.building_median)
        df['building_min'] = df['building_id'].map(self.building_min)
        df['building_max'] = df['building_id'].map(self.building_max)
        df['building_std'] = df['building_id'].map(self.building_std)

        # Weather Data Prep  #####################################################################
        df_weather = prep_weather_data(df_weather)

        # Building MetaData Prep  #####################################################################
        df_building = prep_building_data(df_building)

        # merge data  #####################################################################
        df = pd.merge(df, df_building, how="left", on=["building_id"])
        df = pd.merge(df, df_weather, how='left', on=["site_id", "timestamp"])
        self.df, _ = reduce_mem_usage(df)
        del df, df_weather, df_building
        gc.collect()

        # primary_use  #####################################################################
        if mode == 'train':
            df_group = self.df.groupby('primary_use')['meter_reading']
            self.primary_use_mean = df_group.mean().astype(np.float16)
            self.primary_use_median = df_group.median().astype(np.float16)
            self.primary_use_min = df_group.min().astype(np.float16)
            self.primary_use_max = df_group.max().astype(np.float16)
            self.primary_use_std = df_group.std().astype(np.float16)

        self.df['primary_use_mean'] = self.df['primary_use'].map(self.primary_use_mean)
        self.df['primary_use_median'] = self.df['primary_use'].map(self.primary_use_median)
        self.df['primary_use_min'] = self.df['primary_use'].map(self.primary_use_min)
        self.df['primary_use_max'] = self.df['primary_use'].map(self.primary_use_max)
        self.df['primary_use_std'] = self.df['primary_use'].map(self.primary_use_std)

        # Datetime  #####################################################################
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['month'] = self.df['timestamp'].dt.month.astype(np.uint8)
        self.df['day'] = self.df['timestamp'].dt.day.astype(np.uint8)
        self.df['hour'] = self.df['timestamp'].dt.hour.astype(np.uint8)
        self.df['weekday'] = self.df['timestamp'].dt.weekday.astype(np.uint8)
        # Sort Timestamp  #####################################################################
        self.df = self.df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        del self.df['timestamp']
        gc.collect()

        # LabelEncoder  #####################################################################
        list_cols = ['primary_use']
        if mode == 'train':
            self.ce_oe = ce.OrdinalEncoder(cols=list_cols, handle_unknown='impute')
            self.df = self.ce_oe.fit_transform(self.df)
        elif mode == 'test':
            self.df = self.ce_oe.transform(self.df)

        # Data Type  #####################################################################
        # float32
        cols = self.df.select_dtypes(np.float64).columns
        for c in cols:
            self.df[c] = self.df[c].astype(np.float32)
        # category
        cols = ["site_id", "building_id", "primary_use", "hour", "day", "weekday", "month", "meter"]
        for c in cols:
            self.df[c] = self.df[c].astype('category')

        # sort row_id  #####################################################################
        if mode == 'test':
            self.df = self.df.sort_values(by='row_id').reset_index(drop=True)
