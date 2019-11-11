import numpy as np
import pandas as pd
import gc
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


from .Preprocessing import prep_weather_data, prep_core_data, prep_building_data, reduce_mem_usage, prep_datetime_features


class PreprocessingDataset:
    def __init__(self):
        self.df = None

    def prep(self, df, df_weather, df_building, mode='train'):

        # Core Data Prep  #####################################################################
        if mode == 'train':
            df = prep_core_data(df)

        # Weather Data Prep  #####################################################################
        df_weather = prep_weather_data(df_weather, mode=mode)

        # Building MetaData Prep  #####################################################################
        df_building = prep_building_data(df_building)

        # Merge data  #####################################################################
        df = pd.merge(df, df_building, how="left", on=["building_id"])
        del df_building
        gc.collect()
        df = pd.merge(df, df_weather, how='left', on=["site_id", "timestamp"])
        self.df = reduce_mem_usage(df)
        del df, df_weather
        gc.collect()

        # Prep Datetime  #####################################################################
        self.df = prep_datetime_features(self.df)

        # Sort Timestamp  #####################################################################
        if mode == 'train':
            self.df.sort_values(by=['timestamp', 'building_id'], ascending=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        del self.df['timestamp']
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

        self.df = reduce_mem_usage(self.df)

        # Group feature  #####################################################################
        self.df['building_id_month'] = self.df['building_id'].astype(str) + '_' + self.df['month'].astype(str)
        self.df['building_id_meter_month'] = self.df['building_id'].astype(str) + '_' + \
                                             self.df['meter'].astype(str) + '_' + \
                                             self.df['month'].astype(str)

        # Frequency Encoding  #####################################################################
        cols = ['building_id', 'building_id_month', 'building_id_meter_month']
        for col in cols:
            fq_encode = self.df[col].value_counts().to_dict()
            self.df[col + '_fq_enc'] = self.df[col].map(fq_encode)
            self.df[col + '_fq_enc'] = self.df[col + '_fq_enc'].astype(np.int16)
            del fq_encode
            gc.collect()

        # LabelEncoder  #####################################################################
        list_cols = ['primary_use', 'building_id_month', 'building_id_meter_month']
        temp = self.df[list_cols].copy()
        if mode == 'train':
            self.ce_oe = ce.OrdinalEncoder(cols=list_cols, handle_unknown='impute')
            temp = self.ce_oe.fit_transform(temp)
            temp = temp.astype('float32')
            temp.columns = [s + '_OD_enc' for s in list_cols]
            self.df = pd.concat([self.df, temp], axis=1)
            del temp
            gc.collect()

        elif mode == 'test':
            temp = self.ce_oe.transform(temp)
            temp = temp.astype('float32')
            temp.columns = [s + '_OD_enc' for s in list_cols]
            self.df = pd.concat([self.df, temp], axis=1)
            del temp
            gc.collect()

        # CatBoostEncoder  #####################################################################
        # list_cols = ['primary_use', 'building_id_month', 'building_id_meter_month']
        # temp = self.df[list_cols].copy()
        # if mode == 'train':
        #     self.ce_cat = ce.CatBoostEncoder(cols=list_cols, handle_unknown='impute')
        #     temp = self.ce_cat.fit_transform(temp, self.df['meter_reading'])
        #     temp = temp.astype('float32')
        #     temp.columns = [s + '_CB_enc' for s in list_cols]
        #     self.df = pd.concat([self.df, temp], axis=1)
        #     del temp
        #     gc.collect()
        #
        # elif mode == 'test':
        #     temp = self.ce_cat.transform(temp)
        #     temp = temp.astype('float32')
        #     temp.columns = [s + '_CB_enc' for s in list_cols]
        #     self.df = pd.concat([self.df, temp], axis=1)
        #     del temp
        #     gc.collect()

        # Set_Dtypes  #####################################################################
        self.df = reduce_mem_usage(self.df)

        def set_dtypes(df, cat_cols):
            # category
            for c in cat_cols:
                try:
                    df[c] = df[c].astype('category')
                except:
                    pass

            return df

        cat_cols = ["site_id", "building_id", "primary_use", "hour", "day", "weekday",
                    "month", "meter", 'building_id_month', 'building_id_meter_month']
        self.df = set_dtypes(self.df, cat_cols)
