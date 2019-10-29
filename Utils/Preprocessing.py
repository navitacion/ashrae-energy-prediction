import numpy as np
import pandas as pd


# Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
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

    return df, NAlist


def extract_id_meter(df, building_id, meter):
    temp = df[df['building_id'] == building_id]
    temp = temp[temp['meter'] == meter]
    return temp


# Preprocessing for Core Data
def prep_core_data(df):
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

        return _df

    for _id, meter in zip(df_loss['building_id'], df_loss['meter']):
        df = fill_date(df, _id, meter)

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

            # Date of meter_reading == 0 deals as NaN
            temp.loc[temp['meter_reading'] == 0, 'meter_reading'] = np.nan
            # Use Interpolation for Filling NaN
            temp['meter_reading'] = temp['meter_reading'].interpolate(limit_area='inside', limit=5)
            df.loc[temp.index, 'meter_reading'] = temp.loc[temp.index, 'meter_reading']

    # Dropna    ####################################################################
    df.dropna(inplace=True)

    return df


# Preprocessing Weather Data
def prep_weather_data(df):
    # Drop Features  #####################################################################
    drop_col = ['precip_depth_1_hr', 'sea_level_pressure', 'cloud_coverage']
    df.drop(drop_col, axis=1, inplace=True)

    # Create Features per Site Id  #####################################################################
    # Fillna(Interpolate)
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i]
        temp = temp.sort_values(by='timestamp')

        # Interpolation
        cols = ['air_temperature', 'dew_temperature', 'wind_direction', 'wind_speed']
        for c in cols:
            temp[c] = temp[c].interpolate(limit_direction='both')
            df.loc[temp.index, c] = temp.loc[temp.index, c]

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

    # Wind Direction  #####################################################################
    df.loc[df['wind_direction'] == 65535, 'wind_direction'] = np.nan
    df['wind_direction'] = np.radians(df['wind_direction'])
    df['wind_direction_sin'] = np.sin(df['wind_direction'])
    df['wind_direction_cos'] = np.cos(df['wind_direction'])
    df['wind_direction_tan'] = np.tan(df['wind_direction'])

    df['wind_speed_sin'] = df['wind_speed'] * df['wind_direction_sin']
    df['wind_speed_cos'] = df['wind_speed'] * df['wind_direction_cos']

    # Create Features per Site Id  #####################################################################
    for i in range(df['site_id'].nunique()):
        temp = df[df['site_id'] == i]
        temp = temp.sort_values(by='timestamp')
        # Rolling
        cols = ['air_temperature', 'dew_temperature', 'relative_hummd', 'wind_speed']
        for c in cols:
            for window in [3, 72]:
                # Mean
                colname = '{}_roll_{}_mean'.format(c, window)
                temp[colname] = temp[c].rolling(window).mean()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float32)
                # Sum
                colname = '{}_roll_{}_sum'.format(c, window)
                temp[colname] = temp[c].rolling(window).sum()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float32)
                # Max
                colname = '{}_roll_{}_max'.format(c, window)
                temp[colname] = temp[c].rolling(window).max()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float32)
                # Min
                colname = '{}_roll_{}_min'.format(c, window)
                temp[colname] = temp[c].rolling(window).min()
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]
                df[colname] = df[colname].astype(np.float32)

        # Shift
        cols = ['air_temperature', 'dew_temperature', 'relative_hummd', 'wind_speed']
        for c in cols:
            for period in [1, 24, 48]:
                colname = '{}_shift_{}'.format(c, period)
                shifted = temp[c].shift(periods=period)
                temp[colname] = temp[c] - shifted
                df.loc[temp.index, colname] = temp.loc[temp.index, colname]

    return df


# Preprocessing Building MetaData
def prep_building_data(df):
    # Year Built  #####################################################################
    df['year_built'] = df['year_built'] - 1900 + 1

    # square_feet  #####################################################################
    df['square_feet'] = np.log(df['square_feet'])

    return df
