import numpy as np
import pandas as pd
import datetime
import gc
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

pd.set_option('max_rows', 9999)


# Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
                # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

    return df


def extract_id_meter(df, building_id, meter):
    temp = df[df['building_id'] == building_id].copy()
    temp = temp[temp['meter'] == meter]
    return temp


# Preprocessing for Core Data
def prep_core_data(df, fill_loss_date=True):
    # Log
    df['meter_reading'] = np.log1p(df['meter_reading'].values)

    # Check lossed Date  ####################################################################
    id_list = []
    meter_list = []
    rows_list = []

    for id_ in range(df['building_id'].nunique()):
        for meter in range(4):
            temp = extract_id_meter(df, id_, meter)
            rows = temp.shape[0]
            del temp
            gc.collect()
            if rows not in [0, 8784]:
                id_list.append(id_)
                meter_list.append(meter)
                rows_list.append(rows)

    df_loss = pd.DataFrame({
        'building_id': id_list,
        'meter': meter_list,
        'rows': rows_list
    })
    del id_list, meter_list, rows_list
    gc.collect()

    # Fill dropped Date  ####################################################################
    def fill_date(_df, building_id, meter):
        temp = extract_id_meter(_df, building_id, meter)

        dates_DF = pd.DataFrame(pd.date_range('2016-1-1', periods=366 * 24, freq='H'), columns=['Date'])
        dates_DF['Date'] = dates_DF['Date'].apply(lambda x: x.strftime('%Y-%m-%d %T'))

        temp = pd.merge(temp, dates_DF, how="outer", left_on=['timestamp'], right_on=['Date'])
        del temp['timestamp']
        temp = temp.rename(columns={'Date': 'timestamp'})
        temp['building_id'] = building_id
        temp['meter'] = meter

        temp = temp[temp['meter_reading'].isnull()]
        _df = pd.concat([_df, temp], axis=0, ignore_index=True, sort=True)
        del temp
        gc.collect()

        return _df

    if fill_loss_date:
        for _id, meter in zip(df_loss['building_id'], df_loss['meter']):
            df = fill_date(df, _id, meter)

    del df_loss
    gc.collect()

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
            upper = np.percentile(temp['meter_reading'], 99)
            lower = np.percentile(temp['meter_reading'], 1)
            temp.loc[temp['meter_reading'] > upper, 'meter_reading'] = np.nan
            temp.loc[temp['meter_reading'] < lower, 'meter_reading'] = np.nan

            # Date of meter_reading == 0 deals as NaN
            temp.loc[temp['meter_reading'] == 0, 'meter_reading'] = np.nan
            # Use Interpolation for Filling NaN
            temp['meter_reading'] = temp['meter_reading'].interpolate(limit_area='inside', limit=3)
            df.loc[temp.index, 'meter_reading'] = temp.loc[temp.index, 'meter_reading']

            del temp
            gc.collect()

    # Dropna  ####################################################################
    df.dropna(subset=['meter_reading'], inplace=True)

    return df


# Preprocessing Weather Data
def prep_weather_data(df, mode='train', fill_loss_date=True):
    # Drop Features  #####################################################################
    drop_col = ['precip_depth_1_hr', 'sea_level_pressure', 'cloud_coverage']
    df.drop(drop_col, axis=1, inplace=True)


    # Fill Lossed Date (Only Train)
    if mode == 'train' and fill_loss_date:
        dates_DF = pd.DataFrame(pd.date_range('2016-1-1', periods=366 * 24, freq='H'), columns=['Date'])
        dates_DF['Date'] = dates_DF['Date'].apply(lambda x: x.strftime('%Y-%m-%d %T'))

        for i in range(16):
            temp = df[df['site_id'] == i].copy()
            temp = pd.merge(temp, dates_DF, how="outer", left_on=['timestamp'], right_on=['Date'])
            del temp['timestamp']
            temp = temp.rename(columns={'Date': 'timestamp'})
            temp['site_id'] = i
            df = pd.concat([df, temp], axis=0, ignore_index=True, sort=True)
            df.drop_duplicates(inplace=True)
            del temp
            gc.collect()

        del dates_DF
        gc.collect()

    # Convert GMT  #####################################################################
    # reference  https://www.kaggle.com/patrick0302/locate-cities-according-weather-temperature
    GMT_converter = {0: 4, 1: 0, 2: 7, 3: 4, 4: 7, 5: 0, 6: 4, 7: 4, 8: 4, 9: 5, 10: 7, 11: 4, 12: 0, 13: 5, 14: 4, 15: 4}

    for i in range(16):
        temp = df[df['site_id'] == i].copy()
        temp['timestamp'] = pd.to_datetime(temp['timestamp'].values)
        temp.sort_values(by='timestamp', inplace=True)
        temp['timestamp'] = temp['timestamp'] - datetime.timedelta(hours=GMT_converter[i])
        temp['timestamp'] = temp['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %T'))
        df.loc[temp.index, 'timestamp'] = temp.loc[temp.index, 'timestamp']
        del temp
        gc.collect()

    # Create Features per Site Id  #####################################################################
    # Fillna(Interpolate)
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i].copy()
        temp = temp.sort_values(by='timestamp')

        # Interpolation
        # mixed Linear & Cubic Method  https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116012#latest-667255
        cols = ['air_temperature', 'dew_temperature', 'wind_direction', 'wind_speed']
        for c in cols:
            temp[c] = temp[c].interpolate(method='linear', limit_direction='both')
            df.loc[temp.index, c] = temp.loc[temp.index, c]

        del temp
        gc.collect()

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
        return 0.81 * row['air_temperature'] + 0.01 * row['relative_hummd'] * \
               (0.99 * row['air_temperature'] - 14.3) + 46.3

    df['DI'] = df.apply(disconfort_index, axis=1)

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
    df['beaufort_scale'] = df['beaufort_scale'].astype(np.uint8)

    # Create Features per Site Id  #####################################################################
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i]
        temp = temp.sort_values(by='timestamp')
        # Rolling
        cols = ['air_temperature', 'dew_temperature', 'relative_hummd']
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
        cols = ['air_temperature', 'dew_temperature', 'relative_hummd', 'wind_speed']
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month.astype(np.uint8)
    df['hour'] = df['timestamp'].dt.hour.astype(np.uint8)
    df['dayofweek'] = df['timestamp'].dt.dayofweek.astype(np.uint8)
    df['weekday'] = df['timestamp'].dt.weekday.astype(np.uint8)
    df['weekend'] = 0
    df.loc[(df['weekday'] == 5) | (df['weekday'] == 6), 'weekend'] = 1
    df['weekend'] = df['weekend'].astype(np.uint8)

    # Holiday  #####################################################################
    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    df['is_holiday'] = (
        df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
    df.loc[(df['weekday'] == 5) | (df['weekday'] == 6), 'is_holiday'] = 1
    del us_holidays
    gc.collect()

    return df
