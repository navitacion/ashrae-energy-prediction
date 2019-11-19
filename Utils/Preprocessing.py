import numpy as np
import pandas as pd
import datetime
import gc
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

pd.set_option('max_rows', 9999)


# Recuce_mem
def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    from pandas.api.types import is_categorical_dtype

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    return df


def extract_id_meter(df, building_id, meter):
    temp = df[df['building_id'] == building_id].copy()
    temp = temp[temp['meter'] == meter]
    return temp


# Preprocessing for Core Data
def prep_core_data(df):
    # Log
    df['meter_reading'] = np.log1p(df['meter_reading'].values)

    # Interpolate    ####################################################################
    for _id in range(df['building_id'].nunique()):
        for meter in df['meter'].unique().tolist():
            # Extract by building_id, meter
            temp = extract_id_meter(df, _id, meter)
            temp = temp.sort_values(by='timestamp')

            if temp.empty:
                continue

            # Deal the values between 0 and 0 as Nan
            temp['shifted_past'] = temp['meter_reading'].shift()
            temp['shifted_future'] = temp['meter_reading'].shift(-1)
            drop_rows = temp.query("shifted_past == 0 & shifted_future == 0 & meter_reading > 0")
            df.loc[drop_rows.index, 'meter_reading'] = np.nan

            # Smoothing
            # upper = np.percentile(temp['meter_reading'], 99)
            # lower = np.percentile(temp['meter_reading'], 1)
            # temp.loc[temp['meter_reading'] > upper, 'meter_reading'] = np.nan
            # temp.loc[temp['meter_reading'] < lower, 'meter_reading'] = np.nan

            # Use Interpolation for Filling NaN
            temp['meter_reading'] = temp['meter_reading'].interpolate(limit_area='inside', limit=2)
            df.loc[temp.index, 'meter_reading'] = temp.loc[temp.index, 'meter_reading']

            del temp
            gc.collect()

    # Dropna  ####################################################################
    df.dropna(subset=['meter_reading'], inplace=True)

    return df


# Preprocessing Weather Data
def prep_weather_data(df):
    # Drop Features  #####################################################################
    drop_col = ['precip_depth_1_hr', 'sea_level_pressure', 'cloud_coverage']
    df.drop(drop_col, axis=1, inplace=True)

    # Convert GMT  #####################################################################
    # reference  https://www.kaggle.com/patrick0302/locate-cities-according-weather-temperature
    # GMT_converter = {0: 4, 1: 0, 2: 7, 3: 4, 4: 7, 5: 0, 6: 4, 7: 4, 8: 4, 9: 5, 10: 7, 11: 4, 12: 0, 13: 5, 14: 4, 15: 4}
    #
    # for i in range(16):
    #     temp = df[df['site_id'] == i].copy()
    #     temp['timestamp'] = pd.to_datetime(temp['timestamp'].values)
    #     temp.sort_values(by='timestamp', inplace=True)
    #     temp['timestamp'] = temp['timestamp'] - datetime.timedelta(hours=GMT_converter[i])
    #     temp['timestamp'] = temp['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %T'))
    #     df.loc[temp.index, 'timestamp'] = temp.loc[temp.index, 'timestamp']
    #     del temp
    #     gc.collect()

    # Create Features per Site Id  #####################################################################
    # Fillna(Interpolate)
    # for i in range(df['site_id'].nunique()):
    #     temp = df[df['site_id'] == i].copy()
    #     temp = temp.sort_values(by='timestamp')
    #
    #     # Interpolation
    #     # mixed Linear & Cubic Method  https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116012#latest-667255
    #     cols = ['air_temperature', 'dew_temperature', 'wind_direction', 'wind_speed']
    #     for c in cols:
    #         temp[c] = temp[c].interpolate(method='linear', limit_direction='both')
    #         df.loc[temp.index, c] = temp.loc[temp.index, c]
    #
    #     del temp
    #     gc.collect()

    # relative Hummd  #####################################################################
    # https://soudan1.biglobe.ne.jp/qa5356721.html
    a_temp = df['air_temperature'].values
    d_temp = df['dew_temperature'].values

    def SaturatedWaterVaporPressure(values):
        return 6.11 * 10 ** (7.5 * values / (237.3 + values))

    a_temp = SaturatedWaterVaporPressure(a_temp)
    d_temp = SaturatedWaterVaporPressure(d_temp)

    df['relative_hummd'] = d_temp / a_temp * 100
    del a_temp, d_temp
    gc.collect()

    # Disconfort Index  #####################################################################
    # https://keisan.casio.jp/exec/system/1202883065

    def disconfort_index(row):
        T = row['air_temperature']
        RH = row['relative_hummd']
        return 0.81 * T + 0.01 * RH * (0.99 * T - 14.3) + 46.3

    df['DI'] = df.apply(disconfort_index, axis=1)

    # Apparent Temperature  #####################################################################
    # https://keisan.casio.jp/exec/system/1257417058

    def apparent_temperature(row):
        T = row['air_temperature']
        h = row['relative_hummd']
        A = 1.76 + 1.4 * row['wind_speed'] ** 0.75
        return 37 - (37 - T) / (0.68 - 0.0014 * h + 1/A) - 0.29 * T * (1 - h / 100)

    df['AT'] = df.apply(apparent_temperature, axis=1)

    # WCI  #####################################################################
    # https://www.metsoc.jp/tenki/pdf/2010/2010_01_0057.pdf

    def WCI(row):
        T = row['air_temperature']
        U = row['wind_speed']
        return (33 - T) * (10.45 + 10 * U ** 0.5 - U)

    df['WCI'] = df.apply(WCI, axis=1)

    # Wind Direction  #####################################################################
    df.loc[df['wind_direction'] == 65535, 'wind_direction'] = np.nan
    df.loc[df['wind_direction'] == 360, 'wind_direction'] = 0
    df['wind_direction'] = np.radians(df['wind_direction'])
    df['wind_direction_sin'] = np.sin(df['wind_direction'])
    df['wind_direction_cos'] = np.cos(df['wind_direction'])

    df['wind_speed_sin'] = df['wind_speed'] * df['wind_direction_sin']
    df['wind_speed_cos'] = df['wind_speed'] * df['wind_direction_cos']

    for c in ['wind_speed_sin', 'wind_speed_cos']:
        df[c] = df[c].astype(np.float16)

    # beaufort_scale  #####################################################################
    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9),
                (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

    for item in beaufort:
        df.loc[(df['wind_speed'] >= item[1]) & (df['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]

    # Create Features per Site Id  #####################################################################
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i]
        temp = temp.sort_values(by='timestamp')
        # Rolling
        cols = ['air_temperature', 'dew_temperature', 'relative_hummd', 'AT']
        for c in cols:
            for window in [24, 48, 72, 96]:
                # Mean
                colname = '{}_roll_{}_mean'.format(c, window)
                temp[colname] = temp[c].rolling(window).mean()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)
                # Sum
                colname = '{}_roll_{}_sum'.format(c, window)
                temp[colname] = temp[c].rolling(window).sum()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)
                # Max
                colname = '{}_roll_{}_max'.format(c, window)
                temp[colname] = temp[c].rolling(window).max()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)
                # Min
                colname = '{}_roll_{}_min'.format(c, window)
                temp[colname] = temp[c].rolling(window).min()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)

        # Shift
        cols = ['relative_hummd', 'DI', 'AT']
        for c in cols:
            for period in [24, 48, 72, 96]:
                colname = '{}_shift_{}'.format(c, period)
                shifted = temp[c].shift(periods=period)
                temp[colname] = temp[c] - shifted
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)

        del temp
        gc.collect()

    return df


# Preprocessing Building MetaData
def prep_building_data(df):
    # Year Built  #####################################################################
    df['year_built'] = df['year_built'] - 1900 + 1

    # square_feet  #####################################################################
    df['square_feet'] = np.log(df['square_feet'])

    return df


# Preprocessing Datetime Feature
def prep_datetime_features(df):
    # Datetime  #####################################################################
    df['month'] = df['timestamp'].dt.month.astype(np.uint8)
    df['hour'] = df['timestamp'].dt.hour.astype(np.uint8)
    df['dayofweek'] = df['timestamp'].dt.dayofweek.astype(np.uint8)
    df['weekday'] = df['timestamp'].dt.weekday.astype(np.uint8)

    # Holiday  #####################################################################
    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    df['is_holiday'] = (
        df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
    # df.loc[(df['weekday'] == 5) | (df['weekday'] == 6), 'is_holiday'] = 1
    del us_holidays
    gc.collect()

    return df
