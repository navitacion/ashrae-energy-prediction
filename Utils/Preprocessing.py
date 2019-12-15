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


def prep_weather_data(df):
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

    df['DI'] = df.apply(disconfort_index, axis=1).astype(np.float16)

    # Apparent Temperature  #####################################################################
    # https://keisan.casio.jp/exec/system/1257417058

    def apparent_temperature(row):
        T = row['air_temperature']
        RH = row['relative_hummd']
        A = 1.76 + 1.4 * row['wind_speed'] ** 0.75
        return 37 - (37 - T) / (0.68 - 0.0014 * RH + 1 / A) - 0.29 * T * (1 - RH / 100)

    df['AT'] = df.apply(apparent_temperature, axis=1).astype(np.float16)

    # WCI  #####################################################################
    # https://www.metsoc.jp/tenki/pdf/2010/2010_01_0057.pdf

    def WCI(row):
        T = row['air_temperature']
        U = row['wind_speed']
        return (33 - T) * (10.45 + 10 * U ** 0.5 - U)

    df['WCI'] = df.apply(WCI, axis=1).astype(np.float16)

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

    df['beaufort_scale'] = df['beaufort_scale'].astype(np.float16)

    # Create Features per Site Id  #####################################################################
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i]
        temp = temp.sort_values(by='timestamp')
        # Rolling
        cols = ['relative_hummd', 'DI', 'AT']
        for c in cols:
            for window in [1, 3, 24, 36]:
                # Mean
                colname = '{}_roll_{}_mean'.format(c, window)
                temp[colname] = temp[c].rolling(window).mean()
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
            for period in [1, 3, 24, 36]:
                colname = '{}_shift_{}'.format(c, period)
                shifted = temp[c].shift(periods=period)
                temp[colname] = temp[c] - shifted
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float16)

        del temp
        gc.collect()

    return df


def prepare_data(X, building_data, weather_data, test=False, frac=None):
    """
    Preparing final dataset with all features.
    """

    X = pd.merge(X, building_data, on="building_id", how="left")
    X = pd.merge(X, weather_data, on=["site_id", "timestamp"], how="left")

    if frac is not None:
        X = X.sample(frac=frac, random_state=42)

    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)
    X['square_feet'] = X['square_feet'].astype(np.float16)

    if not test:
        # Data Cleaning
        X['M'] = X['timestamp'].dt.month
        X['D'] = X['timestamp'].dt.day
        X = X.query('not (building_id <= 104 & meter == 0 & M <= 4)')
        X = X.query('not (building_id <= 104 & meter == 0 & M == 5 & D <= 20)')

        # https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks/notebook
        X = X.query('not (meter == 0 & meter_reading == 0)')
        X = X.query('not (building_id == 1099 & meter == 2 & meter_reading > 3e4)')
        X = X.query('not (site_id == 0 & meter == 0 & M <= 4)')
        X = X.query('not (site_id == 0 & meter == 0 & M == 5 & D <= 20)')

        X['M'] = X['M'].astype(np.int16)
        del X['D']

        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)

    gc.collect()

    X.loc[:, "hour"] = X['timestamp'].dt.hour
    X.loc[:, "weekday"] = X['timestamp'].dt.weekday

    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]

    X.drop(drop_features, axis=1, inplace=True)

    # Set dtypes
    categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday"]
    for c in categorical_features:
        X[c] = X[c].astype('category')

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        X.drop("meter_reading", axis=1, inplace=True)
        return X, y
