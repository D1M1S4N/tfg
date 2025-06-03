import pandas as pd
import MetaTrader5 as mt5
import os
from correct_data import correct_candle_data, correct_tick_data

def get_tick_data(symbol, start_date, end_date):
    """
    Fetch tick data for a given symbol between start_date and end_date.
    
    :param symbol: The trading symbol to fetch data for.
    :param start_date: The start date for the data in 'YYYY-MM-DD' format.
    :param end_date: The end date for the data in 'YYYY-MM-DD' format.
    :return: A DataFrame containing the tick data.
    """
    # Ensure MetaTrader 5 is initialized
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return None

    # Convert dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Fetch tick data
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)

    # Shutdown MetaTrader 5 connection
    mt5.shutdown()

    # Convert to DataFrame
    df = pd.DataFrame(ticks)
    
    return df

def get_candle_data(symbol, timeframe, start_date, end_date):
    """
    Fetch OHLCV data for a given symbol and timeframe between start_date and end_date.
    
    :param symbol: The trading symbol to fetch data for.
    :param timeframe: The timeframe for the data (e.g., mt5.TIMEFRAME_M1).
    :param start_date: The start date for the data in 'YYYY-MM-DD' format.
    :param end_date: The end date for the data in 'YYYY-MM-DD' format.
    :return: A DataFrame containing the OHLCV data.
    """
    # Ensure MetaTrader 5 is initialized
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return None

    # Convert dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Fetch OHLCV data
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    # Shutdown MetaTrader 5 connection
    mt5.shutdown()

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    return df

# Save Dataframe to CSV and move it to data folder
def save_to_csv(df, filename):
    """
    Save a DataFrame to a CSV file.
    
    :param df: The DataFrame to save.
    :param filename: The name of the file to save the DataFrame to.
    """
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    
    file_path = os.path.join(data_folder, filename)
    
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    
def load_from_csv(filename) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.
    
    :param filename: The name of the file to load the DataFrame from.
    :return: The loaded DataFrame.
    """
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    
    file_path = os.path.join(data_folder, filename)
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    
    df = pd.read_csv(file_path)
    return df

def timeframe_to_string(timeframe):
    """
    Convierte un valor de timeframe de MetaTrader 5 a una representación de texto.
    
    :param timeframe: Valor numérico del timeframe de MetaTrader 5.
    :return: Representación en texto del timeframe.
    """
    timeframe_dict = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }
    
    return timeframe_dict.get(timeframe, str(timeframe))
    