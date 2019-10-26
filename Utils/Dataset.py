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
        # if mode == 'train':
        #     df = prep_core_data(df)

        # Weather Data Prep  #####################################################################
        # df_weather = prep_weather_data(df_weather)

        # Building MetaData Prep  #####################################################################
        df_building = prep_building_data(df_building)

        # merge data  #####################################################################
        df = pd.merge(df, df_building, how="left", on=["building_id"])
        df = pd.merge(df, df_weather, how='left', on=["site_id", "timestamp"])
        self.df, _ = reduce_mem_usage(df)
        del df, df_weather, df_building
        gc.collect()

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
