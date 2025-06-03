import pandas as pd
from datetime import datetime

def correct_candle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: DataFrame containing OHLCV data.
    :return: Corrected DataFrame.
    """
    df = df.drop(columns = ['real_volume'], axis = 1)
    df = df.rename(columns = {'tick_volume': 'volume'})
    
    df['spread'] = df['spread'] / 100000
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df

def correct_tick_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: DataFrame containing tick data.
    :return: Corrected DataFrame.
    """
    df = df.drop(columns = ['last', 'volume', 'time_msc', 'volume_real'], axis = 1)
    
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['spread'] = df['ask'] - df['bid']
    
    return df

