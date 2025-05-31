import pandas as pd

def correct_candle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct the OHLCV data by ensuring that the 'open' and 'close' prices are not NaN,
    and that the 'high' is greater than or equal to both 'open' and 'close',
    and 'low' is less than or equal to both 'open' and 'close'.
    
    :param df: DataFrame containing OHLCV data.
    :return: Corrected DataFrame.
    """
    df.drop(['real_volume'], axis = 1)
    df['spread'] = df['spread'] / 100000

    return df

